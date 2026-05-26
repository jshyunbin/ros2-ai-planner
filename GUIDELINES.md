# Manipulation Challenge Guidelines

This document has lists of requirements and constraints on the actual challenge execution. 
Strictly follow the guideline. 

## Submission Requirements
All participants must submit their work to the designated directory following these structure rules:
- Package Naming: Each team must load exactly one Docker image named `Image_team_x` (replace x with your assigned team number).
- Documentation: A comprehensive `README_team_x.md` must be attached to the image location, containing:
    - Step-by-step launch procedures.
    - The command to launch the system in "standby mode."

## Technical Environment & Constraints
To ensure a fair and clean competition environment, teams must adhere to the following software constraints:
- Process Flow: Participants must launch their code initially and wait in a "Standby State." The system should only begin processing once it receives an instruction via a specific ROS topic.
- TAs launch the simulator with randomly placed objects. 

    ```ros2 launch manip_challenge ur5_setup.launch.py```
- TAs launch the student program provided by each team.

    ```ros2 launch team_# contest_run.launch.py```
- TAs  will send a random command. 
- No Global Installations: Participants are strictly prohibited from installing libraries to the system root (`/usr/local`, etc.) or the local workspace shared by others.
- Isolation (Docker): All external code and dependencies (e.g., PyTorch, Grounding DINO, Transformers) must be installed and executed within a Docker environment. Launch scripts and nodes must run inside this container. 
- Critical Requirement: Installations must not interfere with or overwrite the dependencies of other teams. 