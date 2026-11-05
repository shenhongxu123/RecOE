"""PIL evaluation loader with explicit, logged recovery of truncated images."""

from threading import RLock
import io
import warnings

from PIL import Image, ImageFile


_pillow_flag_lock = RLock()


def _decode_rgb(path, allow_truncated):
    # DataLoader workers are separate processes. The lock also serializes this
    # loader's calls if used from threads, since Pillow's option is process-wide.
    with _pillow_flag_lock:
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated
            with Image.open(path) as image:
                return image.convert("RGB")
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


def load_rgb_image(path, allow_truncated=False, image_bytes=None):
    """Decode the same sample, retrying only a Pillow truncation error if enabled.

    Ordinary files use strict decoding. Recoverable truncated files are logged;
    no samples are dropped, replaced, or written back. Unrecoverable files fail
    with their path, and Pillow's global flag is restored after every attempt.
    image_bytes lets a disk cache reuse this decoding policy without reading
    the source file; each decoding attempt starts from a fresh byte stream.
    """
    def decode(allow):
        if image_bytes is None:
            return _decode_rgb(path, allow_truncated=allow)
        with io.BytesIO(image_bytes) as source:
            return _decode_rgb(source, allow_truncated=allow)

    try:
        return decode(False)
    except OSError as error:
        truncation_error = str(error).lower().startswith(("image file is truncated", "truncated file read"))
        if not allow_truncated or not truncation_error:
            raise OSError("Could not decode image {!r}: {}".format(str(path), error)) from error
        strict_error = str(error)

    try:
        result = decode(True)
    except OSError as error:
        raise OSError(
            "Could not decode truncated image {!r}, even with recovery enabled: {}".format(str(path), error)
        ) from error

    warnings.warn(
        "Decoded truncated image {!r} with Pillow recovery: {}".format(str(path), strict_error),
        RuntimeWarning,
        stacklevel=2,
    )
    return result
