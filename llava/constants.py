CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Pointer tokens for attention-pointer training
# These mirror GUI-Actor's special tokens so datasets with pointer
# annotations can be reused.
DEFAULT_POINTER_START_TOKEN = "<|pointer_start|>"
DEFAULT_POINTER_END_TOKEN = "<|pointer_end|>"
DEFAULT_POINTER_PAD_TOKEN = "<|pointer_pad|>"
