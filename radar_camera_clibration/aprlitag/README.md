## Convert an AprilTag PNG to SVG

Got the apriltag from here: https://github.com/AprilRobotics/apriltag-imgs/tree/master

Convert a PNG of an AprilTag into an SVG using the provided script. The --size option sets the tag edge length (e.g., 20mm).

```bash
python3 tag_to_svg.py ./tagStandard41h12/tag41_12_00005.png ./svg/tag41_12_00005.svg --size=20mm
```
