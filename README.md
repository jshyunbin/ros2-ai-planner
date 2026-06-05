# ros2-ai-planner

CS477 manipulation challenge를 위한 Dockerized ROS2 AI 플래닝 스택.

## 파이프라인

```text
task_command → 세그멘테이션 (Gemini bbox + SAM2)
             → GraspGen (세그멘테이션된 포인트클라우드에서 파지 포즈)
             → cuRobo (실시간 듀얼 RGBD TSDF 기반 관절 궤적 계획)
             → UR5 실행 (/ur5_controller/follow_joint_trajectory)
             → 목적지 이송 (충돌 미고려 safe-transit_z 경로; 책장은 +x 삽입 / −x 후퇴)
             → 물체 놓기 → 홈 복귀 (충돌 고려)
```

세 AI 단계는 각각 별도의 ROS2 노드로 구현되며, 오케스트레이터가 ROS2 서비스를 통해 조율한다.

**cuRobo가 전체 깊이 파이프라인을 담당한다.** 두 D435 깊이 스트림을 직접 구독해 `base_link` 좌표계에서 GPU 기반 블록-스파스 TSDF/ESDF로 융합하고, 이 맵을 충돌 회피 모션 플래닝에 사용한다.

## 요구사항

- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 호스트에 ROS2 Humble 설치 (`manip_challenge` 실행용)
- NVIDIA GPU (Turing 이상, VRAM ≥ 4 GB)
- NVIDIA 드라이버 ≥ 580 (CUDA 12 지원)
- 호스트에서 `manip_challenge` 실행 중 (Gazebo + ROS2 Humble)

**환경 분리:**

| 위치 | 역할 |
|---|---|
| 호스트 | `manip_challenge`, Gazebo, 베이스 ROS2 그래프 |
| 컨테이너 | AI 플래닝 노드 (자체 ROS2 포함) |

두 환경은 `network_mode: host`로 DDS를 통해 통신한다. 커스텀 서비스 정의는 `src/utils/riro_srvs`에 있다.

## 빠른 시작

```bash
# .env 파일 생성 후 GEMINI_API_KEY 입력
cp .env.example .env

# 이미지 빌드 (첫 빌드 ~20분, 이후 레이어 캐시됨)
docker compose build

# 배포 모드: 전체 파이프라인, UR5 실행, 시각화 없음
docker compose up

# 디버그 모드: 동일 파이프라인 + viser 시각화, ./src 라이브 마운트
docker compose -f docker-compose.yml -f docker-compose.debug.yml up

# 대화형 셸 (디버그 오버라이드 적용)
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --rm ai_planner bash
```

태스크 커맨드 발행 (호스트 또는 소싱된 컨테이너 셸에서):

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String "{data: 'banana'}"
```

cuRobo 플래너 초기화가 완료되기 전에 요청이 도착해도 실패하지 않고 초기화 완료까지 블록된다.

## 런타임 환경 변수

```bash
ROS_DOMAIN_ID=0
ROS_LOCALHOST_ONLY=0
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
FASTDDS_BUILTIN_TRANSPORTS=UDPv4
GEMINI_API_KEY=<키>   # segmentation_service에서 필요
```

## 패키지 구조

`src/` 아래 두 ROS2 패키지:

- `team_8` — 모든 파이프라인 노드
- `utils/riro_srvs` — 커스텀 서비스 정의 (`StringString`, `PlanTrajectory`, …)

### 노드 목록

| 모듈 | 엔트리포인트 | 역할 |
|---|---|---|
| `orchestrator.py` | `orchestrator` | `/task_commands` 구독 → 세그멘테이션 → GraspGen → cuRobo 서비스 순차 호출 → FollowJointTrajectory 실행. pick 이후 목적지 키(`target_goal` 파라미터, 추후 상위 NL 파싱으로 대체)를 cuRobo `goal_name` 모드에 전달해 이송 → 놓기 → 홈 복귀 수행 |
| `segmentation_service.py` | `segmentation_service` | `/segmentation/segment_prompt` 서비스. Gemini로 bbox 추출, SAM2로 마스크 정제, 깊이 역투영 후 세그멘테이션/배경 포인트클라우드 발행 |
| `graspgen_service.py` | `graspgen_service` | `/graspgen/infer` 서비스. 세그멘테이션된 클라우드를 수신해 ZMQ로 GraspGen 서버에 추론 요청, 운동학/충돌 필터링 후 파지 순위 JSON 반환 |
| `curobo_service.py` | `curobo_service` | `/curobo/plan_trajectory` 서비스. cuRobo 플래너 래퍼; pick(접근+파지/들기), 단일 포즈(`grasp_pose`), 또는 `goal_name` 모드 — `place_poses.yml` 키를 충돌 미고려 safe-transit_z 이송(+책장 삽입/후퇴)으로, `home`은 충돌 고려 복귀로 계획 |
| `curobo.py` | — | `CuRobo` 클래스: 듀얼 RGBD TSDF 매핑 + 모션 플래닝 |
| `graspgen_client.py` | — | 독립 GraspGen 추론 서버에 대한 ZMQ 클라이언트 |
| `segmentation_utils.py` | — | 세그멘테이션 헬퍼 (리사이즈, 깊이 역투영, 다운샘플, 중심점, 오버레이) |
| `debug_viz.py` | `debug_viz` | viser 서버 호스팅; 세그멘테이션/배경 클라우드, 파지 포즈, TSDF 복셀 시각화 (디버그 모드 전용) |
| `live_viz_helpers.py` | — | 포인트클라우드/TSDF 시각화 헬퍼 |
| `graspgen_probe.py` | `graspgen_probe` | 독립 디버그 유틸리티 (런타임 파이프라인 외부) |

### 소유권 경계

- **오케스트레이터**: `/joint_states` 구독 및 모든 액션 배포(팔 + 그리퍼) 담당
- **cuRobo**: 깊이 이미지, CameraInfo, TF만 구독; `/joint_states` 구독 안 함 — `update_joint_state()`로 외부에서 주입

## 토픽 & 서비스

### 외부 (manip_challenge / Gazebo)

| 인터페이스 | 타입 | 사용처 |
|---|---|---|
| `/task_commands` | `std_msgs/String` | orchestrator |
| `/wrist_camera/wrist_camera/color/image_raw` | `sensor_msgs/Image` | segmentation_service |
| `/wrist_camera/wrist_camera/depth/color/image_raw` + `camera_info` | Image, CameraInfo | segmentation_service, curobo |
| `/camera/camera/depth/color/image_raw` + `camera_info` | Image, CameraInfo | curobo (오버헤드) |
| `/joint_states` | `sensor_msgs/JointState` | orchestrator, curobo_service |
| `/ur5_controller/follow_joint_trajectory` | action | orchestrator (팔) |
| `/gripper_controller/follow_joint_trajectory` | action | orchestrator (그리퍼) |

### 내부

| 인터페이스 | 타입 | 방향 |
|---|---|---|
| `/segmentation/segment_prompt` | `riro_srvs/StringString` | orchestrator → segmentation_service |
| `/graspgen/segmented_object`, `/graspgen/background` | `sensor_msgs/PointCloud2` | segmentation_service → graspgen_service |
| `/graspgen/infer` | `riro_srvs/StringString` | orchestrator → graspgen_service |
| `/curobo/plan_trajectory` | `riro_srvs/PlanTrajectory` | orchestrator → curobo_service |
| `/graspgen/grasp_poses` | `geometry_msgs/PoseArray` | graspgen_service → debug_viz |
| `/curobo/tsdf_voxels` | `sensor_msgs/PointCloud2` | curobo_service → debug_viz |

## 런치 파일

| 파일 | 역할 |
|---|---|
| `launch/pipeline_common.launch.py` | 공통 노드 그래프 |
| `launch/deploy.launch.py` | 배포 모드 진입점 |
| `launch/debug.launch.py` | 디버그 모드 진입점 |

두 모드 모두 임베디드 GraspGen 서버를 자동 시작하고 모션 실행을 활성화한다. 디버그 모드는 추가로 파지 포즈/TSDF 발행과 `debug_viz` viser 노드(기본 포트 8080)를 활성화한다.

## 개발

### 소스 편집 워크플로

디버그 모드에서 `./src`가 라이브 마운트되므로 Python 노드 편집은 다음 런치에 바로 반영된다. `setup.py`, 엔트리포인트, `*.launch.py` 변경 후에는 컨테이너 내부에서 워크스페이스 리빌드가 필요하다:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
colcon build --symlink-install --packages-select riro_srvs team_8
source install/setup.bash
```

### 의존성 추가

pip 의존성은 `requirements/` 아래 파일별로 관리된다 (`sam2.txt`, `graspgen.txt`, `curobo.txt`, `planner-runtime.txt`). cuRobo는 소스에서 설치해야 한다. 의존성 변경 후에는 이미지 리빌드가 필요하다.

**cuRobo YAML 설정 주의:** `src/team_8/config/ur5_curobo.yml`은 ASCII 전용으로 유지해야 한다. cuRobo의 `load_yaml`이 ASCII 코덱으로 열기 때문에 비ASCII 문자가 있으면 크래시된다.

### Dockerfile 레이어 순서

무거운 레이어(ROS2 apt 패키지, PyTorch CUDA 휠, GraspGen/cuRobo 소스 설치, 모델 다운로드)가 상단에, `COPY src`가 마지막에 위치해 노드 코드 변경 시 최종 레이어만 무효화된다.

## 치트시트

### 호스트 환경 설정

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

```bash
# Gazebo 물리 엔진 언포즈
ros2 service call /unpause_physics std_srvs/srv/Empty "{}"
```

### Docker 컨테이너

```bash
# 배포 모드
docker compose up

# 디버그 모드
docker compose -f docker-compose.yml -f docker-compose.debug.yml up

# 대화형 셸
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --rm ai_planner bash

# 이름 있는 지속 컨테이너
docker compose -f docker-compose.yml -f docker-compose.debug.yml run --name ai_planner_dev --service-ports ai_planner bash

# 실행 중인 컨테이너에 추가 셸
docker exec -it ai_planner_dev bash

# 정리
docker rm -f ai_planner_dev
```

### 컨테이너 셸 초기화

모든 컨테이너 셸에서 실행:

```bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
source install/setup.bash
```

### 태스크 발행

```bash
ros2 topic pub --once /task_commands std_msgs/msg/String "{data: 'banana'}"
```

### 수동 서비스 호출

```bash
ros2 service call /segmentation/segment_prompt riro_srvs/srv/StringString "{data: 'banana'}"
```

```bash
ros2 service call /graspgen/infer std_srvs/srv/Trigger "{}"
```

```bash
# cuRobo 서비스 인터페이스 확인 (보통 orchestrator가 호출)
ros2 interface show riro_srvs/srv/PlanTrajectory
ros2 service type /curobo/plan_trajectory
```

### 런타임 점검

```bash
ros2 node list
ros2 topic list
ros2 service list
ros2 action list
```

```bash
ros2 topic echo /task_commands
ros2 topic echo /joint_states --once
```

### 환경 및 모델 파일 확인

```bash
docker compose run --rm ai_planner env | grep -E 'ROS_DOMAIN_ID|ROS_LOCALHOST_ONLY|RMW_IMPLEMENTATION|FASTDDS_BUILTIN_TRANSPORTS|GEMINI_API_KEY'
```

```bash
ls /opt/GraspGen
ls /opt/GraspGenModels/checkpoints
ls /opt/models/sam2
```

### 아티팩트 경로

```text
./artifacts/segmentation_service/
./artifacts/graspgen_service/
```
