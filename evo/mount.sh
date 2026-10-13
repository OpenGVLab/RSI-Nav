sudo apt-get update
sudo apt-get install libfuse2
proxy_off
cd xxx/

export AWS_ACCESS_KEY_ID="xx"
export AWS_SECRET_ACCESS_KEY="xx"

./s3mount xxx xxx/RSI-Nav/data/trajectory_data --prefix xxx/StreamVLN/trajectory_data/ --endpoint-url xxx --force-path-style --allow-delete --allow-overwrite
./s3mount xxx xxx/RSI-Nav/data/scene/hm3d --prefix xxx/StreamVLN/scene_datasets/hm3d/train/ --endpoint-url xxx --force-path-style --allow-delete --allow-overwrite
./s3mount xxx xxx/RSI-Nav/data/scene/mp3d/ --prefix xxx/StreamVLN/scene_datasets/mp3d/ --endpoint-url xxx --force-path-style --allow-delete --allow-overwrite
./s3mount xxx xxx/cache/ --prefix cache/ --endpoint-url xxx --force-path-style --allow-delete --allow-overwrite

VSI_LOCAL="xxx/RSI-Nav/data/datasets/vsi_590k/VSI-590K"
./s3mount xxx "${VSI_LOCAL}/hypersim/" --prefix hf_data/spatial/VSI-590K/hypersim/ --endpoint-url xxx --force-path-style --allow-delete --allow-overwrite
./s3mount xxx "${VSI_LOCAL}/robotics/" --prefix hf_data/spatial/VSI-590K/robotics/ --endpoint-url xxx --force-path-style --allow-delete --allow-overwrite
./s3mount xxx "${VSI_LOCAL}/ytb_roomtour/" --prefix hf_data/spatial/VSI-590K/ytb_roomtour/ --endpoint-url xxx --force-path-style --allow-delete --allow-overwrite

export AWS_ACCESS_KEY_ID="xx"
export AWS_SECRET_ACCESS_KEY="xx"
BASE_PREFIX="xxx"
BUCKET="xxx"
ENDPOINT="xxx"
VSI_LOCAL="xxx/RSI-Nav/data/datasets/vsi_590k/VSI-590K"
VSI_PREFIX="${BASE_PREFIX}/P~mm_video~en~VSI-590k-video~1.4.0~0.0/multimodal_elements"
./s3mount "${BUCKET}" "${VSI_LOCAL}/adt/" --prefix "${VSI_PREFIX}/adt/" --endpoint-url "${ENDPOINT}" --force-path-style --allow-delete --allow-overwrite
./s3mount "${BUCKET}" "${VSI_LOCAL}/arkitscenes/" --prefix "${VSI_PREFIX}/arkitscenes/" --endpoint-url "${ENDPOINT}" --force-path-style --allow-delete --allow-overwrite
./s3mount "${BUCKET}" "${VSI_LOCAL}/procthor/" --prefix "${VSI_PREFIX}/procthor/" --endpoint-url "${ENDPOINT}" --force-path-style --allow-delete --allow-overwrite
./s3mount "${BUCKET}" "${VSI_LOCAL}/s3dis/" --prefix "${VSI_PREFIX}/s3dis/" --endpoint-url "${ENDPOINT}" --force-path-style --allow-delete --allow-overwrite
./s3mount "${BUCKET}" "${VSI_LOCAL}/scannet/" --prefix "${VSI_PREFIX}/scannet/" --endpoint-url "${ENDPOINT}" --force-path-style --allow-delete --allow-overwrite
./s3mount "${BUCKET}" "${VSI_LOCAL}/scannetppv2/" --prefix "${VSI_PREFIX}/scannetppv2/" --endpoint-url "${ENDPOINT}" --force-path-style --allow-delete --allow-overwrite