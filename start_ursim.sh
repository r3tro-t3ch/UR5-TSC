#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Vishnu Joshi
# Affiliation: CoRIS, Oregon State University
# Email: joshivis@oregonstate.edu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Overview:
# Start URSim (UR's offline simulator: the real UR controller and PolyScope)
# as a UR10e in docker, then power the arm on and release its brakes, so
# ur_rtde talks to 127.0.0.1 exactly like to the real arm. The container is
# kept between runs, so PolyScope keeps its settings, but URSim always boots in
# Local mode: switch it to Remote (needed by ur_rtde) in the PolyScope GUI after
# each start, the script waits for that. Ctrl+C stops the simulator.
# Usage: ./start_ursim.sh [URSim version, default 5.26]

VERSION=${1:-5.26}              # match the real robot's PolyScope version if possible
NAME=ursim_$VERSION             # the simulator container, kept between runs


dashboard() {
    # send one command to URSim's dashboard server and print its reply
    { exec 3<>/dev/tcp/127.0.0.1/29999; } 2>/dev/null || return 1
    read -r -t 5 _ <&3                  # welcome line
    echo "$1" >&3
    read -r -t 5 reply <&3
    exec 3<&-
    echo "${reply%$'\r'}"               # without a trailing carriage return
}

# start the kept simulator in the background, or create it on the first run
docker start "$NAME" > /dev/null 2>&1 || docker run -d --name "$NAME" -e ROBOT_MODEL=UR10 \
    -p 29999:29999 -p 30001-30004:30001-30004 -p 127.0.0.1:5900:5900 -p 127.0.0.1:6080:6080 \
    "universalrobots/ursim_e-series:$VERSION" > /dev/null || exit 1

trap 'docker stop "$NAME" > /dev/null' EXIT    # stop the simulator whenever this script ends

# wait for PolyScope to boot (the dashboard then answers with a robot mode)
echo "booting URSim $VERSION ..."
until dashboard "robotmode" | grep -q "Robotmode"; do sleep 2; done

# ur_rtde needs Remote mode, which can only be switched in the PolyScope GUI
if [ "$(dashboard 'is in remote control')" != "true" ]; then
    echo "switch to Remote: open http://localhost:6080/vnc.html, click Local (top right) > Remote Control"
    echo "(first run only, before that: menu > Settings > System > Remote Control > Enable)"
    until [ "$(dashboard 'is in remote control')" = "true" ]; do sleep 2; done
fi

# power on, wait for idle, release the brakes, wait until the arm is running
dashboard "power on" > /dev/null
until dashboard "robotmode" | grep -q "IDLE"; do sleep 1; done
dashboard "brake release" > /dev/null
until dashboard "robotmode" | grep -q "RUNNING"; do sleep 1; done

echo "URSim ready: ur_rtde at 127.0.0.1, PolyScope at http://localhost:6080/vnc.html, Ctrl+C to stop"
docker wait "$NAME" > /dev/null
