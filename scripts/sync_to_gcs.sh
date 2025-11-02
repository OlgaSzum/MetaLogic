
set -euo pipefail
gcloud storage cp -r outputs/* gs://nonprl-ml/derived/ || true
