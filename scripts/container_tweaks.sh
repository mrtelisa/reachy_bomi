#!/bin/bash
# One-off tweaks to Pollen's stack inside the Reachy Docker container, to be
# re-applied whenever the container is recreated from the image:
#
#   docker cp scripts/container_tweaks.sh reachy2:/tmp/ && docker exec reachy2 bash /tmp/container_tweaks.sh
#
# - Lidar rays: hide the blue laser-scan visualization in Gazebo (the sensor
#   keeps working, /scan is still published). The installed xacro is a symlink
#   to src, so no rebuild is needed.
set -e
XACRO=/home/reachy/reachy_ws/src/mobile_base/zuuu_description/urdf/zuuu.gazebo.xacro
sed -i '/<sensor name="lidar" type="ray">/,/<\/sensor>/ s|<visualize>true</visualize>|<visualize>false</visualize>|' "$XACRO"
grep -n -A2 '<sensor name="lidar"' "$XACRO"
