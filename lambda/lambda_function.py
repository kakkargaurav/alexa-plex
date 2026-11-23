import logging
import os
import random
import time
import boto3
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import SSLError, ConnectionError, Timeout
from difflib import get_close_matches
from urllib.parse import quote
from plexapi.server import PlexServer
from plexapi.base import PlexObject
from ask_sdk_core.skill_builder import CustomSkillBuilder
from ask_sdk_core.dispatch_components import (
    AbstractRequestHandler, AbstractExceptionHandler, AbstractResponseInterceptor
)
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_model.ui import SimpleCard
from ask_sdk_model.interfaces.display import Image, ImageInstance
from ask_sdk_model.interfaces.alexa.presentation.apl import RenderDocumentDirective
from ask_sdk_model.interfaces.audioplayer import (
    PlayDirective, PlayBehavior, AudioItem, Stream, AudioItemMetadata,
    StopDirective
)

def _load_env_file(*names):
    """
    Alexa-hosted skills have no place to set Lambda environment variables, so read KEY=VALUE
    lines from a small file that sits next to this one (.env, or plex.env if your editor won't
    create dotfiles). Real environment variables, if any, always win. No extra packages needed.
    """
    folder = os.path.dirname(os.path.abspath(__file__))
    for name in names:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    if key.startswith("export "):
                        key = key[len("export "):].strip()
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                        value = value[1:-1]
                    os.environ.setdefault(key, value)
            return name
        except Exception:
            return None
    return None


_ENV_FILE_USED = _load_env_file(".env", "plex.env")

# ---------------------------------------------------------------------------
# Configuration
# Set these in a .env file next to this one (see .env.example), or as real environment variables.
#   PLEX_TOKEN=your-plex-token
#   PLEX_URL=https://media.gauravkakkar.au
# There is deliberately no default token: if it is missing the skill says so in the log.
# ---------------------------------------------------------------------------
PLEX_TOKEN = os.environ.get("PLEX_TOKEN", "")
PLEX_URL = os.environ.get("PLEX_URL", "https://media.gauravkakkar.au").rstrip("/")
MUSIC_SECTION_NAME = "Music"

# Alexa-hosted skills get a DynamoDB table provisioned automatically. Its name and region
# are supplied through these built-in environment variables, and its partition key is "id".
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_PERSISTENCE_TABLE_NAME", "PlexAlexaQueue")
DYNAMODB_REGION = os.environ.get("DYNAMODB_PERSISTENCE_REGION")
DYNAMODB_KEY = "id"

# Maximum number of tracks to queue to avoid Lambda timeouts
MAX_TRACKS = 150

# Feature switches. Set both to False to get the plain audio-only behaviour (title and artist
# only, no cover art, no on-screen controls). Handy for finding out which feature breaks something.
ENABLE_ART = True    # album art in the audio metadata
ENABLE_APL = True    # custom Now Playing screen with buttons and seek bar
# Amazon: when a response carries an AudioPlayer.Play directive, shouldEndSession should be true.
# Left unset (or false) Alexa starts the stream and then pauses it to listen for the user.
END_SESSION_AFTER_PLAY = True

# Optional manual spoken-name -> Plex-name mappings (fuzzy matching covers the rest)
ARTIST_MAPPINGS = {
    "sugar free": "Suga Free",
    "austin larolle": "Austin Larold",
    "austin larold": "Austin Larold",
    "austin": "Austin Larold",
    "doctor dre": "Dr. Dre",
    "doctor dray": "Dr. Dre",
    "the doctor": "Dr. Dre",
    "86 love": "86LOVE",
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not PLEX_TOKEN:
    logger.error("PLEX_TOKEN is not set. Add PLEX_TOKEN=... to a .env file next to "
                 "lambda_function.py (see .env.example).")

dynamodb = (boto3.resource('dynamodb', region_name=DYNAMODB_REGION)
            if DYNAMODB_REGION else boto3.resource('dynamodb'))
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

plex = None
MUSIC = None
artist_cache = []
artist_cache_loaded = False


# ---------------------------------------------------------------------------
# Plex connection (lazy, retried on every invocation if it previously failed)
# ---------------------------------------------------------------------------
def connect_plex(budget=5.0):
    """
    Connect directly to the Plex server URL. Returns True when connected.

    Alexa-hosted Lambdas are killed after about 8 seconds, so the whole connection attempt is
    kept inside `budget` seconds. If Plex is slow or unreachable we fail fast and let the skill
    answer with a spoken error instead of timing out.
    """
    global plex, MUSIC
    if plex is not None and MUSIC is not None:
        return True
    if not PLEX_TOKEN:
        logger.error("Cannot connect to Plex: PLEX_TOKEN is not set (see .env.example)")
        return False

    deadline = time.time() + budget
    attempt = 0
    while True:
        attempt += 1
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        step_timeout = min(2.5, remaining / 2)   # server info + library lookup each use this
        started = time.time()
        try:
            logger.info(f"Connecting to Plex at {PLEX_URL} (attempt {attempt}, "
                        f"step timeout {step_timeout:.1f}s)")
            server = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=step_timeout)
            music = server.library.section(MUSIC_SECTION_NAME)
            server._timeout = 6   # allow slower queries (searches, track lists) once connected
            plex, MUSIC = server, music
            logger.info(f"Connected to Plex in {time.time() - started:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Plex connect attempt {attempt} failed after "
                         f"{time.time() - started:.2f}s: {type(e).__name__}: {e}")

    plex = None
    MUSIC = None
    return False


def probe_plex():
    """Diagnostic: log exactly what Lambda sees when it reaches your server."""
    try:
        r = requests.get(f"{PLEX_URL}/identity", timeout=8)
        logger.info(f"Probe status={r.status_code} body={r.text[:200]}")
    except Exception as e:
        logger.error(f"Probe failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_audio_url(track):
    try:
        if track.media and len(track.media) > 0:
            media = track.media[0]
            if media.parts and len(media.parts) > 0:
                part = media.parts[0]
                return f"{PLEX_URL}{part.key}?X-Plex-Token={PLEX_TOKEN}"
        return track.getStreamURL()
    except Exception as e:
        logger.error(f"Error getting audio URL: {e}", exc_info=True)
        return track.getStreamURL()


def get_artist_name(track):
    """Artist name without an extra API call."""
    return getattr(track, 'grandparentTitle', None) or "Unknown Artist"


def check_range_support(url):
    """
    Seeking needs the stream server to honour HTTP Range requests. Ask for a small slice
    and log what comes back (206 = seeking works, 200 = the server ignores Range).
    """
    try:
        r = requests.get(url, headers={"Range": "bytes=1000-1999"}, stream=True, timeout=5)
        status = r.status_code
        info = (f"status={status} accept-ranges={r.headers.get('Accept-Ranges')} "
                f"content-range={r.headers.get('Content-Range')} "
                f"content-length={r.headers.get('Content-Length')} "
                f"content-type={r.headers.get('Content-Type')}")
        r.close()
        if status == 206:
            logger.info(f"Range check OK: {info}")
        else:
            logger.warning(f"Range check FAILED - server is not honouring Range, so seeking will "
                           f"restart the song: {info}")
    except Exception as e:
        logger.warning(f"Range check error: {type(e).__name__}: {e}")


def get_art(track):
    """Album/artist artwork for screen devices (Echo Show etc.), or None."""
    if not ENABLE_ART:
        return None
    thumb = (getattr(track, 'parentThumb', None) or
             getattr(track, 'grandparentThumb', None) or
             getattr(track, 'thumb', None))
    if not thumb:
        return None
    # Ask Plex for a 512x512 JPEG so the image is a size/format Alexa accepts
    url = (f"{PLEX_URL}/photo/:/transcode?width=512&height=512&minSize=1&upscale=1"
           f"&url={quote(thumb, safe='')}&X-Plex-Token={PLEX_TOKEN}")
    return Image(
        content_description=track.title,
        sources=[ImageInstance(url=url, width_pixels=512, height_pixels=512)]
    )


def build_play_directive(track, offset=0, behavior=PlayBehavior.REPLACE_ALL,
                         expected_previous_token=None):
    art = get_art(track)
    # Amazon: if audioItem.metadata is sent it must have title, subtitle, art AND backgroundImage
    metadata = AudioItemMetadata(
        title=track.title,
        subtitle=get_artist_name(track),
        art=art,
        background_image=art
    ) if art else None
    stream = Stream(
        token=str(track.ratingKey),
        url=get_audio_url(track),
        offset_in_milliseconds=offset,
        expected_previous_token=expected_previous_token
    )
    audio_item = AudioItem(stream=stream, metadata=metadata)
    return PlayDirective(play_behavior=behavior, audio_item=audio_item)


# Material-style 24x24 icon paths used for the on-screen controls
APL_ICONS = {
    "iconPlay": "M8 5v14l11-7z",
    "iconPause": "M6 19h4V5H6v14zm8-14v14h4V5h-4z",
    "iconNext": "M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z",
    "iconPrevious": "M6 6h2v12H6zm3.5 6l8.5 6V6z",
    "iconShuffle": ("M10.59 9.17L5.41 4 4 5.41l5.17 5.17 1.42-1.41zM14.5 4l2.04 2.04L4 18.59 "
                    "5.41 20 17.96 7.46 20 9.5V4h-5.5zm.33 9.41l-1.41 1.41 3.13 3.13L14.5 "
                    "20H20v-5.5l-2.04 2.04-3.13-3.13z"),
}

SHUFFLE_ON_COLOR = "#1ED760"       # green
SHUFFLE_ON_GLOW = "#1ED76040"      # translucent green halo behind the icon


def _icon(icon, color=None):
    graphic = {
        "type": "VectorGraphic",
        "source": icon,
        "scale": "best-fit",
        "width": "56dp",
        "height": "56dp"
    }
    if color:
        graphic["fillColor"] = color
    return {
        "type": "Container",
        "width": "88dp",
        "height": "88dp",
        "alignItems": "center",
        "justifyContent": "center",
        "items": [graphic]
    }


def _apl_button(icon, on_press, display=None):
    button = {"type": "TouchWrapper", "onPress": on_press, "item": _icon(icon)}
    if display:
        button["display"] = display
    return button


def _send(*arguments):
    return {"type": "SendEvent", "arguments": list(arguments)}


def _set(prop, value):
    return {"type": "SetValue", "property": prop, "value": value}


APL_SHUFFLE_BUTTON = {
    "type": "TouchWrapper",
    # flip the colour immediately, the skill then confirms the real state
    "onPress": [_set("shuffleOn", "${!shuffleOn}"), _send("shuffle")],
    "item": {
        "type": "Frame",
        "borderRadius": "44dp",
        "borderWidth": "2dp",
        "borderColor": "${shuffleOn ? '%s' : 'transparent'}" % SHUFFLE_ON_COLOR,
        "backgroundColor": "${shuffleOn ? '%s' : 'transparent'}" % SHUFFLE_ON_GLOW,
        "item": _icon("iconShuffle", "${shuffleOn ? '%s' : '#FFFFFF'}" % SHUFFLE_ON_COLOR)
    }
}

SEEK_BAR_SEGMENTS = 30


def _time_text(var):
    return ("${Math.floor(%s/60000)}:${Math.floor(%s/1000)%%60 < 10 ? '0' : ''}"
            "${Math.floor(%s/1000)%%60}") % (var, var, var)


def _seek_segment(i):
    """One tappable slice of the progress bar; tapping it seeks to that point in the song."""
    centre = (i + 0.5) / SEEK_BAR_SEGMENTS
    target = "${Math.floor(total * %.4f)}" % centre
    return {
        "type": "TouchWrapper",
        "grow": 1,
        "shrink": 1,
        "height": "40dp",
        "paddingLeft": "1dp",
        "paddingRight": "1dp",
        "onPress": [_set("pos", target), _set("playing", True), _send("seek", target)],
        "item": {
            "type": "Frame",
            "width": "100%",
            "height": "100%",
            "borderRadius": "3dp",
            "backgroundColor": "${pos >= total * %.4f ? '#FFFFFF' : '#4A4A4A'}" % centre
        }
    }


APL_SEEK_BAR = {
    "type": "Container",
    "width": "${viewport.width > viewport.height ? '40vw' : '80vw'}",
    "paddingTop": "12dp",
    "display": "${total > 0 ? 'normal' : 'none'}",
    "items": [
        {
            "type": "Container",
            "direction": "row",
            "width": "100%",
            "items": [_seek_segment(i) for i in range(SEEK_BAR_SEGMENTS)]
        },
        {
            "type": "Container",
            "direction": "row",
            "width": "100%",
            "justifyContent": "spaceBetween",
            "paddingTop": "4dp",
            "items": [
                {"type": "Text", "text": _time_text("pos"), "color": "#CCCCCC", "fontSize": "18dp"},
                {"type": "Text", "text": _time_text("total"), "color": "#CCCCCC", "fontSize": "18dp"}
            ]
        }
    ]
}

APL_NOW_PLAYING_DOCUMENT = {
    "type": "APL",
    "version": "2023.3",
    "graphics": {
        name: {
            "type": "AVG",
            "version": "1.1",
            "width": 24,
            "height": 24,
            "parameters": [{"name": "fillColor", "type": "color", "default": "#FFFFFF"}],
            "items": [{"type": "path", "pathData": path, "fill": "${fillColor}"}]
        } for name, path in APL_ICONS.items()
    },
    "mainTemplate": {
        "parameters": ["payload"],
        "items": [{
            "type": "Frame",
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": "#000000",
            "item": {
                "type": "Container",
                "width": "100vw",
                "height": "100vh",
                "direction": "${viewport.width > viewport.height ? 'row' : 'column'}",
                "alignItems": "center",
                "justifyContent": "center",
                # State shared by the slider and the buttons
                "bind": [
                    {"name": "pos", "value": "${payload.data.position}"},
                    {"name": "total", "value": "${payload.data.duration}"},
                    {"name": "playing", "value": "${payload.data.playing}"},
                    {"name": "shuffleOn", "value": "${payload.data.shuffle}"}
                ],
                # Advance the bar once a second while playing (resynced on every redraw)
                "handleTick": [{
                    "when": "${playing && pos < total}",
                    "minimumDelay": 1000,
                    "commands": [_set("pos", "${Math.min(pos + 1000, total)}")]
                }],
                "items": [
                    {
                        "type": "Image",
                        "source": "${payload.data.art}",
                        "width": "${viewport.width > viewport.height ? '60vh' : '55vw'}",
                        "height": "${viewport.width > viewport.height ? '60vh' : '55vw'}",
                        "scale": "best-fill",
                        "borderRadius": "12dp"
                    },
                    {
                        "type": "Container",
                        "alignItems": "center",
                        "maxWidth": "90vw",
                        "paddingLeft": "${viewport.width > viewport.height ? '32dp' : '0dp'}",
                        "paddingTop": "${viewport.width > viewport.height ? '0dp' : '16dp'}",
                        "items": [
                            {
                                "type": "Text",
                                "text": "${payload.data.title}",
                                "color": "#FFFFFF",
                                "fontSize": "30dp",
                                "fontWeight": "bold",
                                "maxLines": 2,
                                "textAlign": "center"
                            },
                            {
                                "type": "Text",
                                "text": "${payload.data.subtitle}",
                                "color": "#CCCCCC",
                                "fontSize": "22dp",
                                "maxLines": 1,
                                "textAlign": "center"
                            },
                            APL_SEEK_BAR,
                            {
                                "type": "Container",
                                "direction": "row",
                                "paddingTop": "16dp",
                                "items": [
                                    APL_SHUFFLE_BUTTON,
                                    _apl_button("iconPrevious", [_send("previous")]),
                                    # Pause shows while playing, Play shows while paused
                                    _apl_button("iconPause", [_set("playing", False), _send("pause")],
                                                display="${playing ? 'normal' : 'none'}"),
                                    _apl_button("iconPlay", [_set("playing", True), _send("play")],
                                                display="${playing ? 'none' : 'normal'}"),
                                    _apl_button("iconNext", [_send("next")]),
                                ]
                            }
                        ]
                    }
                ]
            }
        }]
    }
}


def supports_apl(handler_input):
    if not ENABLE_APL:
        return False
    try:
        interfaces = handler_input.request_envelope.context.system.device.supported_interfaces
        return getattr(interfaces, 'alexa_presentation_apl', None) is not None
    except Exception:
        return False


def build_apl_directive(track, position_ms=0, playing=True, shuffle=False):
    art = get_art(track)
    art_url = art.sources[0].url if art else ""
    return RenderDocumentDirective(
        token="nowPlaying",
        document=APL_NOW_PLAYING_DOCUMENT,
        datasources={"data": {
            "art": art_url,
            "title": track.title,
            "subtitle": get_artist_name(track),
            "position": int(position_ms or 0),
            "duration": int(getattr(track, 'duration', 0) or 0),
            "playing": bool(playing),
            "shuffle": bool(shuffle)
        }}
    )


def get_shuffle_state(handler_input):
    """Whether shuffle is currently on for this user's saved queue."""
    try:
        queue_data = get_queue(get_user_id(handler_input))
        return bool(queue_data and queue_data.get('shuffle'))
    except Exception:
        return False


def add_apl_screen(handler_input, response_builder, track, position_ms=0, playing=True, shuffle=None):
    """Show title, artist, art, progress bar and controls on screen devices that support APL."""
    if supports_apl(handler_input):
        if shuffle is None:
            shuffle = get_shuffle_state(handler_input)
        response_builder.add_directive(build_apl_directive(track, position_ms, playing, shuffle))


def get_user_id(handler_input):
    return handler_input.request_envelope.context.system.user.user_id


def get_track_by_key(rating_key):
    try:
        if not connect_plex():
            return None
        return plex.fetchItem(int(rating_key))
    except Exception as e:
        logger.error(f"Error fetching track {rating_key}: {e}")
        return None


# ---------------------------------------------------------------------------
# DynamoDB queue helpers
# ---------------------------------------------------------------------------
def serialize_track(track, artist_name=None):
    try:
        return {
            'key': int(track.ratingKey),
            'title': track.title,
            'artist': artist_name or get_artist_name(track)
        }
    except Exception as e:
        logger.warning(f"Error serializing track: {e}")
        return None


def serialize_tracks(tracks, artist_name=None):
    serialized = []
    for track in tracks:
        result = serialize_track(track, artist_name)
        if result:
            serialized.append(result)
    return serialized


def save_queue(user_id, tracks, current_index=0, shuffle=False, serialized_tracks=None):
    try:
        track_data = serialized_tracks if serialized_tracks is not None else serialize_tracks(tracks)
        item = {
            DYNAMODB_KEY: user_id,
            'tracks': track_data,
            'current_index': int(current_index),
            'shuffle': shuffle,
            'playback_offset': 0
        }
        table.put_item(Item=item)
        logger.info(f"Saved queue for {user_id}: {len(track_data)} tracks, shuffle={shuffle}")
    except Exception as e:
        logger.error(f"Error saving queue to DynamoDB: {e}", exc_info=True)


def get_queue(user_id):
    try:
        response = table.get_item(Key={DYNAMODB_KEY: user_id})
        return response.get('Item')
    except Exception as e:
        logger.error(f"Error getting queue from DynamoDB: {e}", exc_info=True)
        return None


def _update_field(user_id, field, value):
    try:
        table.update_item(
            Key={DYNAMODB_KEY: user_id},
            UpdateExpression=f'SET {field} = :v',
            ExpressionAttributeValues={':v': int(value)}
        )
    except Exception as e:
        logger.error(f"Error updating {field} in DynamoDB: {e}", exc_info=True)


def update_queue_index(user_id, new_index):
    _update_field(user_id, 'current_index', new_index)


def update_retry_count(user_id, count):
    _update_field(user_id, 'retry_count', count)


def update_playback_offset(user_id, offset):
    _update_field(user_id, 'playback_offset', offset)


def clear_playback_offset(user_id):
    try:
        table.update_item(
            Key={DYNAMODB_KEY: user_id},
            UpdateExpression='REMOVE playback_offset'
        )
    except Exception as e:
        logger.error(f"Error clearing playback offset: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Plex fetch helpers
# ---------------------------------------------------------------------------
def plex_api_call_with_retry(func, *args, max_retries=3, **kwargs):
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (SSLError, ConnectionError, Timeout) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Plex call failed (attempt {attempt + 1}/{max_retries}): "
                               f"{type(e).__name__}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Plex call failed after {max_retries} attempts: {e}")
    raise last_exception


def fetch_playlist_tracks_paginated(playlist, target_count, page_size=50):
    """Fetch random pages of a large playlist instead of the whole thing."""
    try:
        total_size = getattr(playlist, 'leafCount', None)
        if total_size is None:
            playlist.reload()
            total_size = getattr(playlist, 'leafCount', 0)

        if total_size <= target_count:
            return list(playlist.items())

        total_pages = (total_size + page_size - 1) // page_size
        pages_needed = (target_count + page_size - 1) // page_size
        pages_to_fetch = sorted(random.sample(range(total_pages), min(pages_needed, total_pages)))

        all_tracks = []
        server = playlist._server
        items_endpoint = f"{playlist.key}/items"

        for page_num in pages_to_fetch:
            start_index = page_num * page_size
            paginated_url = (f"{items_endpoint}?X-Plex-Container-Start={start_index}"
                             f"&X-Plex-Container-Size={page_size}")
            try:
                response = server.query(paginated_url)
                all_tracks.extend(PlexObject.findItems(server, response))
            except (SSLError, ConnectionError, Timeout) as e:
                logger.warning(f"Connection error on page {page_num}, retrying: {e}")
                time.sleep(1)
                try:
                    response = server.query(paginated_url)
                    all_tracks.extend(PlexObject.findItems(server, response))
                except Exception as retry_error:
                    logger.warning(f"Retry failed for page {page_num}: {retry_error}")
                    continue
            except Exception as page_error:
                logger.warning(f"Error fetching page {page_num}: {page_error}")
                continue

            if len(all_tracks) >= target_count:
                break

        return all_tracks
    except Exception as e:
        logger.warning(f"Paginated fetch failed, falling back to full fetch: {e}")
        return list(playlist.items())


def load_artist_cache():
    global artist_cache, artist_cache_loaded
    if artist_cache_loaded:
        return artist_cache
    if not connect_plex():
        logger.error("Plex not available, cannot load artist cache")
        return []
    try:
        artists = plex_api_call_with_retry(MUSIC.searchArtists)
        artist_cache = [a.title for a in artists]
        artist_cache_loaded = True
        logger.info(f"Loaded {len(artist_cache)} artists into cache")
        return artist_cache
    except Exception as e:
        logger.error(f"Error loading artist cache: {e}", exc_info=True)
        return []


def fuzzy_match_artist(spoken_name):
    manual_match = ARTIST_MAPPINGS.get(spoken_name.lower())
    if manual_match:
        return manual_match

    artists = load_artist_cache()
    if not artists:
        return spoken_name

    lower_map = {a.lower(): a for a in artists}
    matches = get_close_matches(spoken_name.lower(), list(lower_map), n=1, cutoff=0.6)
    if matches:
        logger.info(f"Fuzzy match: '{spoken_name}' -> '{lower_map[matches[0]]}'")
        return lower_map[matches[0]]
    return spoken_name


# Intent name -> what it plays (None = work it out from the words spoken)
PLAY_INTENTS = {
    "PlayMusicIntent": None,
    "PlayPlaylistIntent": "playlist",
    "PlayArtistIntent": "artist",
    "PlayAlbumIntent": "album",
    "PlayTrackIntent": "track",
    "ShuffleMusicIntent": None,
}

_KIND_PREFIXES = [
    ("music by the artist ", "artist"), ("some music by ", "artist"), ("some songs by ", "artist"),
    ("music by ", "artist"), ("songs by ", "artist"), ("the artist ", "artist"), ("artist ", "artist"),
    ("my playlist ", "playlist"), ("the playlist ", "playlist"), ("playlist ", "playlist"),
    ("the album ", "album"), ("album ", "album"),
    ("the song ", "track"), ("song ", "track"), ("the track ", "track"), ("track ", "track"),
]


def parse_query(query, kind):
    """
    Turn free-form speech such as 'playlist satinder' or 'Abbey Road by The Beatles' into
    (kind, name, artist_hint). kind stays None when it can't be told from the words.
    """
    q = query.strip()
    low = q.lower()

    if kind is None:
        for prefix, k in _KIND_PREFIXES:
            if low.startswith(prefix):
                kind, q = k, q[len(prefix):].strip()
                break
        else:
            if low.endswith(" playlist"):
                kind, q = "playlist", q[:-len(" playlist")].strip()

    if kind == "playlist":
        for lead in ("my ", "the "):
            if q.lower().startswith(lead):
                q = q[len(lead):].strip()

    artist_hint = None
    if kind in ("album", "track", None) and " by " in q.lower():
        idx = q.lower().rindex(" by ")
        artist_hint = q[idx + 4:].strip() or None
        q = q[:idx].strip()

    return kind, q, artist_hint


def resolve_ambiguous(name):
    """Work out whether a bare name is a playlist, artist, album or track (in that order)."""
    low = name.lower()

    for playlist in plex.playlists():
        if low in playlist.title.lower():
            return "playlist"

    artists = load_artist_cache()
    lower_map = {a.lower(): a for a in artists}
    if low in lower_map or any(low in a for a in lower_map) or \
            get_close_matches(low, list(lower_map), n=1, cutoff=0.75):
        return "artist"

    if plex_api_call_with_retry(MUSIC.searchAlbums, title=name):
        return "album"
    if plex_api_call_with_retry(MUSIC.searchTracks, title=name):
        return "track"
    return None


def prefer_artist(results, artist_hint):
    """If the user said 'X by Y', prefer results whose artist matches Y."""
    if not artist_hint or not results:
        return results
    hint = artist_hint.lower()
    matched = [r for r in results
               if hint in (getattr(r, 'grandparentTitle', None) or
                           getattr(r, 'parentTitle', None) or "").lower()]
    return matched or results


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------
class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        try:
            if not connect_plex():
                speech_text = ("Warning: I couldn't connect to your Plex server. "
                               "Please check your configuration and try again later.")
                return handler_input.response_builder.speak(speech_text).set_card(
                    SimpleCard("Plex Connection Error", speech_text)).set_should_end_session(True).response

            speech_text = "Plex Music is ready. You can ask me to play music from your Plex server."
            reprompt_text = "You can say, for example, play music by Queen, or play the album Abbey Road."
            return handler_input.response_builder.speak(speech_text).ask(reprompt_text).set_card(
                SimpleCard("Plex Music", speech_text)).response
        except Exception as e:
            logger.error(f"Error in LaunchRequestHandler: {e}", exc_info=True)
            return handler_input.response_builder.speak(
                "Sorry, there was an error starting the skill.").set_should_end_session(True).response


class PlayMusicIntentHandler(AbstractRequestHandler):
    """Handles PlayMusicIntent, PlayPlaylist/Artist/Album/TrackIntent and ShuffleMusicIntent."""

    def can_handle(self, handler_input):
        if not is_request_type("IntentRequest")(handler_input):
            return False
        return handler_input.request_envelope.request.intent.name in PLAY_INTENTS

    @staticmethod
    def _slot_value(slots, name):
        if slots and name in slots:
            slot = slots[name]
            if slot and getattr(slot, 'value', None):
                return slot.value
        return None

    def handle(self, handler_input):
        speech_text = "Sorry, something went wrong."
        rb = handler_input.response_builder

        try:
            if not connect_plex():
                return rb.speak("I couldn't reach your Plex server just now. "
                                "Please try again in a moment.").set_should_end_session(True).response

            intent = handler_input.request_envelope.request.intent
            slots = intent.slots
            user_id = get_user_id(handler_input)

            should_shuffle = intent.name == "ShuffleMusicIntent"

            artist_name = self._slot_value(slots, "artist")
            album_name = self._slot_value(slots, "album")
            track_name = self._slot_value(slots, "track")
            playlist_name = self._slot_value(slots, "playlist")
            artist_hint = None

            # Free-form "query" slot (AMAZON.SearchQuery) used by the newer interaction model
            query = self._slot_value(slots, "query")
            if query:
                kind, name, artist_hint = parse_query(query, PLAY_INTENTS.get(intent.name))
                if kind is None:
                    kind = resolve_ambiguous(name)
                logger.info(f"Query '{query}' -> kind={kind}, name='{name}', artist_hint={artist_hint}")
                if kind == "playlist":
                    playlist_name = name
                elif kind == "artist":
                    artist_name = name
                elif kind == "album":
                    album_name = name
                elif kind == "track":
                    track_name = name
                else:
                    return rb.speak(f"I couldn't find anything called {name} in your library."
                                    ).set_should_end_session(True).response

            if artist_name:
                artist_name = fuzzy_match_artist(artist_name)

            logger.info(f"Request - Artist: {artist_name}, Album: {album_name}, "
                        f"Track: {track_name}, Playlist: {playlist_name}, Shuffle: {should_shuffle}")

            tracks_to_play = []
            cached_artist_name = None

            if playlist_name:
                try:
                    matching_playlist = None
                    for playlist in plex.playlists():
                        if playlist_name.lower() in playlist.title.lower():
                            matching_playlist = playlist
                            break

                    if not matching_playlist:
                        return rb.speak(f"I couldn't find a playlist named {playlist_name}."
                                        ).set_should_end_session(True).response

                    all_tracks = fetch_playlist_tracks_paginated(matching_playlist, MAX_TRACKS)
                    total_tracks = getattr(matching_playlist, 'leafCount', len(all_tracks))
                    tracks_to_play = all_tracks[:MAX_TRACKS]

                    if total_tracks > MAX_TRACKS:
                        speech_text = (f"Playing {len(tracks_to_play)} randomly selected tracks from "
                                       f"playlist {matching_playlist.title}, which has {total_tracks} total tracks.")
                    else:
                        speech_text = f"Playing playlist {matching_playlist.title}."
                    if should_shuffle:
                        speech_text = f"Shuffling playlist {matching_playlist.title}."
                except Exception as e:
                    logger.error(f"Error searching playlists: {e}", exc_info=True)
                    return rb.speak(f"I had trouble searching for the playlist {playlist_name}."
                                    ).set_should_end_session(True).response

            elif track_name:
                results = prefer_artist(
                    plex_api_call_with_retry(MUSIC.searchTracks, title=track_name), artist_hint)
                if not results:
                    return rb.speak(f"I couldn't find the track {track_name}."
                                    ).set_should_end_session(True).response
                tracks_to_play = [results[0]]
                speech_text = f"Playing {results[0].title} by {get_artist_name(results[0])}."

            elif album_name:
                results = prefer_artist(
                    plex_api_call_with_retry(MUSIC.searchAlbums, title=album_name), artist_hint)
                if not results:
                    return rb.speak(f"I couldn't find the album {album_name}."
                                    ).set_should_end_session(True).response
                album = results[0]
                tracks_to_play = list(plex_api_call_with_retry(album.tracks))[:MAX_TRACKS]
                cached_artist_name = getattr(album, 'parentTitle', None)
                speech_text = f"Playing the album {album.title}."
                if should_shuffle:
                    speech_text = f"Shuffling the album {album.title}."

            elif artist_name:
                results = plex_api_call_with_retry(MUSIC.searchArtists, title=artist_name)
                if not results:
                    return rb.speak(f"I couldn't find the artist {artist_name}."
                                    ).set_should_end_session(True).response
                artist = results[0]
                cached_artist_name = artist.title

                all_artist_tracks = list(plex_api_call_with_retry(artist.tracks))
                total_tracks = len(all_artist_tracks)
                tracks_to_process = (all_artist_tracks[:MAX_TRACKS * 3]
                                     if total_tracks > MAX_TRACKS else all_artist_tracks)

                seen_keys = set()
                unique_tracks = []
                for t in tracks_to_process:
                    if t.ratingKey not in seen_keys:
                        seen_keys.add(t.ratingKey)
                        unique_tracks.append(t)
                        if len(unique_tracks) >= MAX_TRACKS * 2:
                            break

                tracks_to_play = unique_tracks[:MAX_TRACKS]
                speech_text = f"Playing music by {artist.title}."
                if should_shuffle:
                    speech_text = f"Shuffling music by {artist.title}."
            else:
                logger.warning(f"No usable slot values received for {intent.name}: {slots}")
                speech_text = ("I didn't catch what to play. Try saying, play playlist, followed by "
                               "the name, or play music by an artist.")
                return rb.speak(speech_text).ask(speech_text).response

            if not tracks_to_play:
                return rb.speak("I couldn't find any tracks to play."
                                ).set_should_end_session(True).response

            if should_shuffle and len(tracks_to_play) > 1:
                tracks_to_play = list(tracks_to_play)
                random.shuffle(tracks_to_play)

            serialized = serialize_tracks(tracks_to_play, artist_name=cached_artist_name)
            save_queue(user_id, tracks_to_play, 0, should_shuffle, serialized_tracks=serialized)

            first_track = tracks_to_play[0]
            logger.info(f"Playing track: {first_track.title}")
            play_directive = build_play_directive(first_track)

            rb.speak(speech_text).add_directive(play_directive).set_card(
                SimpleCard("Now Playing", f"{first_track.title} by {get_artist_name(first_track)}"))
            add_apl_screen(handler_input, rb, first_track, shuffle=should_shuffle)
            return rb.response

        except Exception as e:
            logger.error(f"Error in PlayMusicIntentHandler: {e}", exc_info=True)
            speech_text = "Sorry, I had trouble processing your request."

        return rb.speak(speech_text).set_should_end_session(True).response


class _QueueNavigationHandler(AbstractRequestHandler):
    """Shared logic for Next / Previous."""
    INTENT = None
    STEP = 1
    END_MESSAGE = ""
    LOAD_ERROR = ""
    GENERIC_ERROR = ""

    def can_handle(self, handler_input):
        return is_intent_name(self.INTENT)(handler_input)

    def handle(self, handler_input):
        rb = handler_input.response_builder
        try:
            user_id = get_user_id(handler_input)
            queue_data = get_queue(user_id)

            if not queue_data or not queue_data.get('tracks'):
                return rb.speak("There's no queue. Please play something first.").response

            tracks = queue_data['tracks']
            target_index = int(queue_data.get('current_index', 0)) + self.STEP

            if target_index < 0 or target_index >= len(tracks):
                return rb.speak(self.END_MESSAGE).response

            # PlaybackStarted updates the stored index
            track = get_track_by_key(tracks[target_index]['key'])
            if not track:
                return rb.speak(self.LOAD_ERROR).response

            rb.add_directive(build_play_directive(track))
            add_apl_screen(handler_input, rb, track)
            return rb.response
        except Exception as e:
            logger.error(f"Error in {self.INTENT} handler: {e}", exc_info=True)
            return rb.speak(self.GENERIC_ERROR).response


class NextIntentHandler(_QueueNavigationHandler):
    INTENT = "AMAZON.NextIntent"
    STEP = 1
    END_MESSAGE = "You've reached the end of the queue."
    LOAD_ERROR = "Sorry, I couldn't load the next track."
    GENERIC_ERROR = "Sorry, I had trouble skipping to the next track."


class PreviousIntentHandler(_QueueNavigationHandler):
    INTENT = "AMAZON.PreviousIntent"
    STEP = -1
    END_MESSAGE = "You're at the beginning of the queue."
    LOAD_ERROR = "Sorry, I couldn't load the previous track."
    GENERIC_ERROR = "Sorry, I had trouble going back to the previous track."


class PlaybackStartedHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("AudioPlayer.PlaybackStarted")(handler_input)

    def handle(self, handler_input):
        started_offset = getattr(handler_input.request_envelope.request, 'offset_in_milliseconds', None)
        logger.info(f"Playback started at offset {started_offset}ms")
        try:
            current_token = handler_input.request_envelope.request.token
            user_id = get_user_id(handler_input)
            queue_data = get_queue(user_id)

            if queue_data and queue_data.get('tracks'):
                tracks = queue_data['tracks']

                if queue_data.get('retry_count', 0) > 0:
                    update_retry_count(user_id, 0)

                if queue_data.get('playback_offset'):
                    clear_playback_offset(user_id)

                for index, track_info in enumerate(tracks):
                    if str(track_info['key']) == str(current_token):
                        if index != int(queue_data.get('current_index', 0)):
                            update_queue_index(user_id, index)
                        break
        except Exception as e:
            logger.error(f"Error in PlaybackStarted: {e}", exc_info=True)

        return handler_input.response_builder.response


class PlaybackNearlyFinishedHandler(AbstractRequestHandler):
    """Auto-queue the next track."""

    def can_handle(self, handler_input):
        return is_request_type("AudioPlayer.PlaybackNearlyFinished")(handler_input)

    def handle(self, handler_input):
        rb = handler_input.response_builder
        try:
            current_token = handler_input.request_envelope.request.token
            user_id = get_user_id(handler_input)
            logger.info(f"Playback nearly finished for token {current_token}")
            queue_data = get_queue(user_id)

            if not queue_data or not queue_data.get('tracks'):
                logger.info("No queue found, not enqueuing")
                return rb.response

            stored_index = int(queue_data.get('current_index', 0))
            tracks = queue_data['tracks']

            actual_current_index = None
            for index, track_info in enumerate(tracks):
                if str(track_info['key']) == str(current_token):
                    actual_current_index = index
                    break

            if actual_current_index is None:
                actual_current_index = stored_index
            elif actual_current_index != stored_index:
                update_queue_index(user_id, actual_current_index)

            next_index = actual_current_index + 1
            if next_index >= len(tracks):
                logger.info("Reached end of queue, not enqueuing")
                return rb.response

            track = get_track_by_key(tracks[next_index]['key'])
            if not track:
                logger.error("Could not fetch next track from Plex")
                return rb.response

            play_directive = build_play_directive(
                track,
                behavior=PlayBehavior.ENQUEUE,
                expected_previous_token=current_token
            )
            logger.info(f"Enqueued next track '{track.title}' (index {next_index}) after token {current_token}")
            return rb.add_directive(play_directive).response

        except Exception as e:
            logger.error(f"Error in PlaybackNearlyFinishedHandler: {e}", exc_info=True)
            return rb.response


class PlaybackFinishedHandler(AbstractRequestHandler):
    """
    Alexa only allows Stop or ClearQueue in reply to PlaybackFinished, so the next track must
    already have been queued by the PlaybackNearlyFinished handler. Nothing to do here but log.
    """

    def can_handle(self, handler_input):
        return is_request_type("AudioPlayer.PlaybackFinished")(handler_input)

    def handle(self, handler_input):
        token = getattr(handler_input.request_envelope.request, 'token', None)
        logger.info(f"Playback finished for token {token}")
        return handler_input.response_builder.response


class PlaybackStoppedHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("AudioPlayer.PlaybackStopped")(handler_input)

    def handle(self, handler_input):
        try:
            user_id = get_user_id(handler_input)
            request = handler_input.request_envelope.request
            offset = getattr(request, 'offset_in_milliseconds', 0)
            logger.info(f"Playback stopped at offset {offset}ms for token {getattr(request, 'token', None)}")
            if offset and offset > 0:
                update_playback_offset(user_id, offset)
        except Exception as e:
            logger.error(f"Error saving playback offset: {e}", exc_info=True)
        return handler_input.response_builder.response


class PlaybackFailedHandler(AbstractRequestHandler):
    MAX_PLAYBACK_RETRIES = 5

    def can_handle(self, handler_input):
        return is_request_type("AudioPlayer.PlaybackFailed")(handler_input)

    def handle(self, handler_input):
        rb = handler_input.response_builder
        error = handler_input.request_envelope.request.error
        logger.error(f"Playback failed: {error}")

        if error and hasattr(error, 'type') and 'SERVICE_UNAVAILABLE' in str(error.type):
            try:
                user_id = get_user_id(handler_input)
                queue_data = get_queue(user_id)

                if queue_data and queue_data.get('tracks'):
                    current_index = int(queue_data.get('current_index', 0))
                    retry_count = int(queue_data.get('retry_count', 0))
                    tracks = queue_data['tracks']

                    if retry_count < self.MAX_PLAYBACK_RETRIES:
                        update_retry_count(user_id, retry_count + 1)
                        track = get_track_by_key(tracks[current_index]['key'])
                        if track:
                            return rb.add_directive(build_play_directive(track)).response
                        logger.error("Could not fetch current track for retry")
                    else:
                        update_retry_count(user_id, 0)
                        next_index = current_index + 1
                        if next_index < len(tracks):
                            update_queue_index(user_id, next_index)
                            track = get_track_by_key(tracks[next_index]['key'])
                            if track:
                                return rb.add_directive(build_play_directive(track)).response
                            logger.error("Could not fetch next track")
            except Exception as e:
                logger.error(f"Error during playback retry: {e}", exc_info=True)

        return rb.response


class PauseIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.PauseIntent")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.add_directive(StopDirective()).response


class ResumeIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.ResumeIntent")(handler_input)

    def handle(self, handler_input):
        rb = handler_input.response_builder
        try:
            user_id = get_user_id(handler_input)
            queue_data = get_queue(user_id)

            if not queue_data or not queue_data.get('tracks'):
                return rb.speak("There's nothing to resume. Please ask me to play something.").response

            current_index = int(queue_data.get('current_index', 0))
            tracks = queue_data['tracks']
            playback_offset = int(queue_data.get('playback_offset', 0))

            if current_index >= len(tracks):
                return rb.speak("There's nothing to resume. Please ask me to play something.").response

            track = get_track_by_key(tracks[current_index]['key'])
            if not track:
                return rb.speak("Sorry, I couldn't load the track to resume.").response

            return rb.add_directive(build_play_directive(track, offset=playback_offset)).response
        except Exception as e:
            logger.error(f"Error in ResumeIntentHandler: {e}", exc_info=True)
            return rb.speak("Sorry, I had trouble resuming playback.").response


class ShuffleOnIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.ShuffleOnIntent")(handler_input)

    def handle(self, handler_input):
        rb = handler_input.response_builder
        try:
            user_id = get_user_id(handler_input)
            queue_data = get_queue(user_id)

            if not queue_data or not queue_data.get('tracks'):
                return rb.speak("There's no queue to shuffle. Please play something first.").response

            tracks = list(queue_data['tracks'])
            current_index = int(queue_data.get('current_index', 0))
            current_track = tracks[current_index]

            random.shuffle(tracks)
            tracks = [t for t in tracks if t['key'] != current_track['key']]
            tracks.insert(0, current_track)

            queue_data['tracks'] = tracks
            queue_data['current_index'] = 0
            queue_data['shuffle'] = True
            table.put_item(Item=queue_data)

            return rb.speak("Shuffle is now on.").response
        except Exception as e:
            logger.error(f"Error in ShuffleOnIntentHandler: {e}", exc_info=True)
            return rb.speak("Sorry, I had trouble turning on shuffle.").response


class ShuffleOffIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.ShuffleOffIntent")(handler_input)

    def handle(self, handler_input):
        rb = handler_input.response_builder
        try:
            user_id = get_user_id(handler_input)
            queue_data = get_queue(user_id)
            if queue_data:
                queue_data['shuffle'] = False
                table.put_item(Item=queue_data)
            return rb.speak("Shuffle is now off. Note: The current queue order won't change, "
                            "but new content will play in order.").response
        except Exception as e:
            logger.error(f"Error in ShuffleOffIntentHandler: {e}", exc_info=True)
            return rb.speak("Sorry, I had trouble turning off shuffle.").response


class WhatsPlayingIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("WhatsPlayingIntent")(handler_input)

    def handle(self, handler_input):
        rb = handler_input.response_builder
        try:
            user_id = get_user_id(handler_input)
            queue_data = get_queue(user_id)

            if not queue_data or not queue_data.get('tracks'):
                return rb.speak("Nothing is currently playing.").response

            current_index = int(queue_data.get('current_index', 0))
            tracks = queue_data['tracks']

            if current_index < len(tracks):
                current_track = tracks[current_index]
                title = current_track.get('title', 'Unknown')
                artist = current_track.get('artist', 'Unknown Artist')
                return rb.speak(f"You're listening to {title} by {artist}.").set_card(
                    SimpleCard("Now Playing", f"{title}\nby {artist}")).response

            return rb.speak("I couldn't determine what's currently playing.").response
        except Exception as e:
            logger.error(f"Error in WhatsPlayingIntentHandler: {e}", exc_info=True)
            return rb.speak("Sorry, I had trouble getting the current track information.").response


class RateSongIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("RateSongIntent")(handler_input)

    def handle(self, handler_input):
        rb = handler_input.response_builder
        try:
            user_id = get_user_id(handler_input)
            queue_data = get_queue(user_id)

            if not queue_data or not queue_data.get('tracks'):
                return rb.speak("Nothing is currently playing. Please play a song first.").response

            slots = handler_input.request_envelope.request.intent.slots
            rating_value = None

            if slots and "rating" in slots:
                slot = slots["rating"]
                if slot and getattr(slot, 'value', None):
                    try:
                        rating_value = float(slot.value)
                    except ValueError:
                        return rb.speak("I didn't understand the rating. Please say a number from 1 to 5.").response

            if rating_value is None:
                return rb.speak("Please specify a rating from 1 to 5 stars.").response

            if rating_value < 1 or rating_value > 5:
                return rb.speak("Please provide a rating between 1 and 5 stars.").response

            current_index = int(queue_data.get('current_index', 0))
            tracks = queue_data['tracks']

            if current_index >= len(tracks):
                return rb.speak("I couldn't determine what's currently playing.").response

            info = tracks[current_index]
            track_title = info.get('title', 'Unknown')
            track = get_track_by_key(info['key'])

            if not track:
                return rb.speak("Sorry, I couldn't access the current track to rate it.").response

            try:
                track.rate(rating_value * 2.0)  # Plex uses a 0-10 scale
                stars = int(rating_value)
                speech_text = f"I've rated {track_title} {stars} star{'s' if stars != 1 else ''}."
                return rb.speak(speech_text).set_card(
                    SimpleCard("Song Rated", f"{track_title}\nRating: {stars} stars")).response
            except Exception as e:
                logger.error(f"Error rating track: {e}", exc_info=True)
                return rb.speak("Sorry, I had trouble setting the rating in Plex.").response

        except Exception as e:
            logger.error(f"Error in RateSongIntentHandler: {e}", exc_info=True)
            return rb.speak("Sorry, I had trouble rating the song.").response


class APLUserEventHandler(AbstractRequestHandler):
    """Handles touches on the on-screen controls (APL SendEvent)."""

    def can_handle(self, handler_input):
        return is_request_type("Alexa.Presentation.APL.UserEvent")(handler_input)

    def handle(self, handler_input):
        try:
            arguments = handler_input.request_envelope.request.arguments or []
            action = str(arguments[0]).lower() if arguments else ""
            logger.info(f"APL UserEvent: {action} arguments={arguments}")

            if action == "next":
                return NextIntentHandler().handle(handler_input)
            if action == "previous":
                return PreviousIntentHandler().handle(handler_input)
            if action == "play":
                return ResumeIntentHandler().handle(handler_input)
            if action == "pause":
                return PauseIntentHandler().handle(handler_input)
            if action == "seek":
                return self._seek(handler_input, arguments)
            if action == "shuffle":
                queue_data = get_queue(get_user_id(handler_input))
                if queue_data and queue_data.get('shuffle'):
                    return ShuffleOffIntentHandler().handle(handler_input)
                return ShuffleOnIntentHandler().handle(handler_input)
        except Exception as e:
            logger.error(f"Error in APLUserEventHandler: {e}", exc_info=True)
        return handler_input.response_builder.response


    @staticmethod
    def _seek(handler_input, arguments):
        rb = handler_input.response_builder

        # Which track is playing: from the device's playback state, else from the saved queue
        audio = handler_input.request_envelope.context.audio_player
        track_key = audio.token if audio and audio.token else None
        if not track_key:
            queue_data = get_queue(get_user_id(handler_input))
            if queue_data and queue_data.get('tracks'):
                index = int(queue_data.get('current_index', 0))
                if 0 <= index < len(queue_data['tracks']):
                    track_key = queue_data['tracks'][index]['key']
        if not track_key:
            logger.warning("Seek ignored: no current track")
            return rb.response

        track = get_track_by_key(track_key)
        if not track:
            logger.warning(f"Seek ignored: could not load track {track_key}")
            return rb.response

        # The slider sends its thumb position twice; use the first usable value
        position = None
        for raw in arguments[1:]:
            try:
                value = int(float(raw))
            except (TypeError, ValueError):
                continue
            if value > 0:
                position = value
                break
        if position is None:
            logger.warning(f"Seek ignored: no usable position in {arguments}")
            return rb.response

        duration = int(getattr(track, 'duration', 0) or 0)
        # Stay clear of the very end so the track doesn't finish instantly
        position = max(0, min(position, max(duration - 1000, 0)))
        logger.info(f"Seeking '{track.title}' to {position}ms of {duration}ms")
        check_range_support(get_audio_url(track))

        rb.add_directive(build_play_directive(track, offset=position))
        add_apl_screen(handler_input, rb, track, position, playing=True)
        return rb.response


class FallbackIntentHandler(AbstractRequestHandler):
    """Catches speech that doesn't match any intent, instead of Alexa guessing the closest one."""

    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        logger.info("FallbackIntent: utterance did not match any intent")
        speech_text = ("Sorry, I didn't catch that. You can say play playlist, followed by a name, "
                       "or play music by an artist.")
        return handler_input.response_builder.speak(speech_text).ask(speech_text).response


class ExceptionEncounteredHandler(AbstractRequestHandler):
    """
    Alexa sends System.ExceptionEncountered when it rejected one of our responses (for example an
    invalid directive). Log the reason; the response to this request must be empty.
    """

    def can_handle(self, handler_input):
        return is_request_type("System.ExceptionEncountered")(handler_input)

    def handle(self, handler_input):
        request = handler_input.request_envelope.request
        logger.error(f"System.ExceptionEncountered: error={getattr(request, 'error', None)} "
                     f"cause={getattr(request, 'cause', None)}")
        return handler_input.response_builder.response


class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        speech_text = ("You can ask me to play a song, album, artist, or playlist from your Plex server. "
                       "For example, say play The Beatles, play the album Abbey Road, or play playlist "
                       "Favorites. You can also say shuffle, next, previous, pause, or stop to control "
                       "playback. You can rate the current song by saying rate this song 3 stars.")
        return handler_input.response_builder.speak(speech_text).ask(speech_text).response


class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (is_intent_name("AMAZON.CancelIntent")(handler_input) or
                is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        return handler_input.response_builder.speak("Goodbye!").add_directive(
            StopDirective()).set_should_end_session(True).response


class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        request = handler_input.request_envelope.request
        logger.info(f"Session ended. reason={getattr(request, 'reason', None)} "
                    f"error={getattr(request, 'error', None)}")
        return handler_input.response_builder.response


class AllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(f"Unhandled exception: {exception}", exc_info=True)

        # AudioPlayer, PlaybackController and System requests can't carry speech
        request_type = getattr(handler_input.request_envelope.request, 'object_type', '') or ''
        if request_type.startswith(("AudioPlayer.", "PlaybackController.", "System.")):
            return handler_input.response_builder.response

        return handler_input.response_builder.speak(
            "Sorry, there was an error. Please try again.").set_should_end_session(True).response


# ---------------------------------------------------------------------------
# Skill wiring
# ---------------------------------------------------------------------------
class KeepNowPlayingScreenInterceptor(AbstractResponseInterceptor):
    """
    If audio is playing or paused and a voice/touch request gets a reply without a screen
    (e.g. "I couldn't find the artist"), re-send the Now Playing screen so the display
    doesn't go blank. Only applies to intent and APL touch requests, because AudioPlayer
    events cannot return screen directives.
    """

    def process(self, handler_input, response):
        try:
            request = handler_input.request_envelope.request
            request_type = request.object_type
            if request_type not in ("IntentRequest", "Alexa.Presentation.APL.UserEvent"):
                return
            if request_type == "IntentRequest" and request.intent.name in (
                    "AMAZON.StopIntent", "AMAZON.CancelIntent"):
                return
            if not supports_apl(handler_input):
                return

            audio = handler_input.request_envelope.context.audio_player
            if not audio or not audio.token:
                return
            activity = getattr(audio.player_activity, 'value', audio.player_activity)
            if activity not in ("PLAYING", "PAUSED", "BUFFERING"):
                return

            directives = response.directives or []
            if any(isinstance(d, RenderDocumentDirective) for d in directives):
                return

            track = get_track_by_key(audio.token)
            if not track:
                return
            # Work out what the player will be doing after this response
            offset = getattr(audio, 'offset_in_milliseconds', 0) or 0
            playing = (activity == "PLAYING")
            for d in directives:
                if isinstance(d, StopDirective):
                    playing = False
                elif isinstance(d, PlayDirective):
                    playing = True
                    offset = d.audio_item.stream.offset_in_milliseconds or 0
            response.directives = directives + [
                build_apl_directive(track, offset, playing, get_shuffle_state(handler_input))]
            logger.info("Re-sent Now Playing screen")
        except Exception as e:
            logger.error(f"Error in KeepNowPlayingScreenInterceptor: {e}", exc_info=True)


class EndSessionAfterPlayInterceptor(AbstractResponseInterceptor):
    """Set shouldEndSession=true on voice/touch responses that start audio (see END_SESSION_AFTER_PLAY)."""

    def process(self, handler_input, response):
        try:
            if not END_SESSION_AFTER_PLAY:
                return
            request_type = handler_input.request_envelope.request.object_type
            if request_type not in ("IntentRequest", "LaunchRequest", "Alexa.Presentation.APL.UserEvent"):
                return
            if any(isinstance(d, PlayDirective) for d in (response.directives or [])) \
                    and response.should_end_session is None:
                response.should_end_session = True
        except Exception as e:
            logger.error(f"Error in EndSessionAfterPlayInterceptor: {e}", exc_info=True)


sb = CustomSkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(PlayMusicIntentHandler())
sb.add_request_handler(APLUserEventHandler())
sb.add_request_handler(NextIntentHandler())
sb.add_request_handler(PreviousIntentHandler())
sb.add_request_handler(PlaybackStartedHandler())
sb.add_request_handler(PlaybackNearlyFinishedHandler())
sb.add_request_handler(PlaybackFinishedHandler())
sb.add_request_handler(PlaybackStoppedHandler())
sb.add_request_handler(PlaybackFailedHandler())
sb.add_request_handler(PauseIntentHandler())
sb.add_request_handler(ResumeIntentHandler())
sb.add_request_handler(ShuffleOnIntentHandler())
sb.add_request_handler(ShuffleOffIntentHandler())
sb.add_request_handler(WhatsPlayingIntentHandler())
sb.add_request_handler(RateSongIntentHandler())
sb.add_request_handler(ExceptionEncounteredHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

sb.add_global_response_interceptor(KeepNowPlayingScreenInterceptor())
sb.add_global_response_interceptor(EndSessionAfterPlayInterceptor())
sb.add_exception_handler(AllExceptionHandler())

handler = sb.lambda_handler()


def lambda_handler(event, context):
    request = event.get('request', {})
    logger.info(f"Request type: {request.get('type', 'Unknown')}")
    if request.get('type') == 'IntentRequest':
        intent = request.get('intent', {})
        logger.info(f"Intent: {intent.get('name')} slots: {intent.get('slots', {})}")
    response = handler(event, context)
    # Log the exact response Alexa receives (token masked) to debug "problem with the response"
    logger.info("Response: " + (str(response).replace(PLEX_TOKEN, "***") if PLEX_TOKEN else str(response)))
    return response