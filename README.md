# ros2-ai-planner

CS477 manipulation challenge를 위한 Dockerized ROS2 AI 플래닝 스택.

## 파이프라인

```text
task_command → 세그멘테이션 (Gemini bbox + SAM2)
             → GraspGenX (세그멘테이션된 포인트클라우드에서 파지 포즈 생성)
             → cuRobo (실시간 듀얼 RGBD TSDF 기반 관절 궤적 계획)
             → UR5 실행 (/ur5_controller/follow_joint_trajectory)
```

세 AI 단계는 각각 별도의 ROS2 노드로 구현되며, 오케스트레이터가 ROS2 서비스를 통해 조율한다.

**cuRobo가 전체 깊이 파이프라인을 담당한다.** 두 D435 깊이 스트림을 직접 구독해 `base_link` 좌표계에서 GPU 기반 블록-스파스 TSDF/ESDF로 융합하고, 이 맵을 충돌 회피 모션 플래닝에 사용한다.

**GraspGenX**는 ZMQ로 통신하는 독립 추론 서버로 동작한다. `graspgen_ext` Docker named volume에 의존성이 캐싱되어 재시작 시 재다운로드가 생략된다.

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

## 파이프라인 동작 상세

### 세그멘테이션 및 멀티-스캔 재시도

오케스트레이터는 홈 포즈에서 세그멘테이션을 시도한다. 객체 감지 신뢰도가 기준 미달이거나 포인트클라우드가 부족하면 `PIPELINE_SCAN_POSES`에 정의된 추가 뷰포인트로 팔을 이동한 후 재시도한다. 각 스캔 이후 TSDF를 리셋(`/curobo/reset_map`)하여 새 뷰포인트의 깊이 데이터가 깨끗하게 반영된다.

### GraspGenX 파지 필터링 및 순위 결정

GraspGenX가 생성한 파지 후보들은 두 단계로 필터링된다:

1. **경성 틸트 필터** (`PIPELINE_GRASPGEN_MAX_TILT_DEG`, 기본 45°): 수직 하향 방향에서 이 각도 이상 기울어진 파지는 제거된다.
2. **합성 점수 순위** (`PIPELINE_GRASPGEN_VERTICALITY_WEIGHT`, 기본 0.85):
   ```
   score = w × alignment + (1-w) × confidence
   ```
   수직 정렬(alignment) 가중치를 높여 GraspGenX의 신뢰도만 높은 비스듬한 파지보다 수직 파지를 우선시한다.

### Direct-to-Grasp 플래닝

Pre-grasp standoff(접근 중간 포즈)를 사용하지 않고 cuRobo goalset 플래너로 파지 포즈에 직접 도달하는 단일 궤적을 계획한다. TSDF에 의한 충돌 회피가 동시에 적용된다.

```
현재 관절 상태 → [cuRobo goalset 플래너] → 파지 포즈 (직접)
```

### 계층적 goalset 플래닝 (Tiered Planning)

cuRobo goalset 플래너는 관절 공간 이동 비용을 최소화하므로 수직 파지보다 비스듬한 파지를 선호할 수 있다. 이를 방지하기 위해 두 단계로 플래닝을 시도한다:

| 단계 | 후보 집합 | 조건 |
|---|---|---|
| Tier-0 | `approach_z ≥ PIPELINE_CUROBO_TIER0_MIN_APPROACH_Z` (기본 0.85, ≈ 32° 이내) | 수직에 가까운 파지만 시도 |
| Fallback | 모든 후보 | Tier-0 실패 시 |
| 최종 Fallback | 모든 후보 + TSDF 클리어 | 모든 TSDF 기반 시도 실패 시 |

### 바닥 충돌 방지 (두 성분 z-클램프)

그리퍼가 테이블을 뚫거나 떨리는 현상을 방지하기 위해 각 파지 후보의 tool0 z 위치에 두 성분 최소값을 적용한다:

```
min_tool_z_tilt = floor_z + approach_z × fingertip_len + approach_z × descent_margin
min_tool_z_abs  = floor_z + gripper_body_clearance
effective_min   = max(min_tool_z_tilt, min_tool_z_abs)
```

- **틸트 스케일 성분**: 파지 각도에 따라 손가락 끝이 바닥에 닿지 않도록 보호
- **절대값 성분** (`PIPELINE_CUROBO_GRIPPER_BODY_CLEARANCE`, 기본 0.06 m): 비스듬한 파지에서 그리퍼 본체(손바닥, 링크)가 테이블 모서리를 침범하지 않도록 보호

## 패키지 구조

`src/` 아래 두 ROS2 패키지:

- `team_8` — 모든 파이프라인 노드
- `utils/riro_srvs` — 커스텀 서비스 정의 (`StringString`, `PlanTrajectory`, …)

### 노드 목록

| 모듈 | 엔트리포인트 | 역할 |
|---|---|---|
| `orchestrator.py` | `orchestrator` | `/task_commands` 구독 → 세그멘테이션 → GraspGenX → cuRobo 서비스 순차 호출 → FollowJointTrajectory 실행. 멀티-스캔 재시도 및 자동 루프(auto-loop) 지원 |
| `segmentation_service.py` | `segmentation_service` | `/segmentation/segment_prompt` 서비스. Gemini로 bbox 추출, SAM2로 마스크 정제, 깊이 역투영 후 세그멘테이션/배경 포인트클라우드 발행 |
| `graspgen_service.py` | `graspgen_service` | `/graspgen/infer` 서비스. 세그멘테이션된 클라우드를 수신해 ZMQ로 GraspGenX 서버에 추론 요청, 틸트 필터링·순위 결정 후 파지 포즈 JSON 반환 |
| `curobo_service.py` | `curobo_service` | `/curobo/plan_trajectory` 서비스. cuRobo 플래너 래퍼; pick(direct-to-grasp + 들기) 또는 단일 포즈(place/home) 계획 |
| `curobo.py` | — | `CuRobo` 클래스: 듀얼 RGBD TSDF 매핑 + 계층적 goalset 모션 플래닝 + 바닥 z-클램프 |
| `graspgen_client.py` | — | GraspGenX 추론 서버에 대한 ZMQ 클라이언트 |
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
| `/curobo/reset_map` | `std_msgs/Empty` | orchestrator → curobo (스캔 이동 후 TSDF 리셋) |
| `/curobo/ready` | `std_msgs/Bool` (latched) | curobo → orchestrator (플래너 초기화 완료 신호) |
| `/graspgen/grasp_poses` | `geometry_msgs/PoseArray` | graspgen_service → debug_viz |
| `/curobo/tsdf_voxels` | `sensor_msgs/PointCloud2` | curobo_service → debug_viz |

## 런치 파일

| 파일 | 역할 |
|---|---|
| `launch/pipeline_common.launch.py` | 공통 노드 그래프 |
| `launch/deploy.launch.py` | 배포 모드 진입점 |
| `launch/debug.launch.py` | 디버그 모드 진입점 |

두 모드 모두 임베디드 GraspGenX 서버를 자동 시작하고 모션 실행을 활성화한다. 디버그 모드는 추가로 파지 포즈/TSDF 발행과 `debug_viz` viser 노드(기본 포트 8080)를 활성화한다.

## 주요 환경 변수 레퍼런스

`docker-compose.debug.yml`에서 조정 가능한 주요 파라미터:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PIPELINE_FLOOR_Z` | `-0.089` | 테이블 표면 z 좌표 (base_link 기준, m) |
| `PIPELINE_SCAN_POSES` | (홈 포즈만) | 세그멘테이션 재시도 뷰포인트. `;`로 구분된 관절각 목록 |
| `PIPELINE_GEMINI_MIN_CONFIDENCE` | `0.8` | Gemini 감지 신뢰도 최소값 |
| `PIPELINE_GRASPGEN_MAX_TILT_DEG` | `45` | 파지 틸트 경성 필터 (도). 이 이상 기울어진 파지 제거 |
| `PIPELINE_GRASPGEN_VERTICALITY_WEIGHT` | `0.85` | 수직 정렬 가중치 (`score = w×align + (1-w)×conf`) |
| `PIPELINE_CUROBO_TIER0_MIN_APPROACH_Z` | `0.85` | Tier-0 플래닝 threshold (cos 틸트). ≈32° 이내 파지 우선 시도 |
| `PIPELINE_CUROBO_DESCENT_MARGIN` | `0.03` | 틸트 스케일 z-클램프 여유 (m) |
| `PIPELINE_CUROBO_GRIPPER_BODY_CLEARANCE` | `0.06` | 그리퍼 본체 절대 바닥 여유 (m) |
| `PIPELINE_CUROBO_GRASP_CLOSE_EXTRA_M` | `0.100` | tool0를 TCP에서 후퇴시키는 거리. 0.1034 - 0.100 = +3.4 mm 여유 |
| `PIPELINE_CUROBO_PICK_LIFT_OFFSET` | `0.15` | 파지 후 들어올리는 높이 (m) |
| `PIPELINE_GRIPPER_CLOSE_POSITION` | `0.6` | 그리퍼 닫힘 위치 (0.0=완전 열림, 0.8=완전 닫힘) |
| `PIPELINE_GRASP_CLOSE_NUDGE_MAX_JOINT_DELTA_RAD` | `0` | pre-close nudge 비활성화 (바닥 침범 방지) |
| `PIPELINE_CUROBO_CORRIDOR_RADIUS` | `0.06` | 객체 상단 충돌 corridor 반경 (m) |
| `PIPELINE_CUROBO_GRASP_SPHERE_CARVE_RADIUS` | `0.15` | 파지 지점 TSDF sphere carve 반경 (m) |

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

GraspGenX의 외부 의존성은 `graspgen_ext` named volume에 캐싱된다. 볼륨이 존재하면 컨테이너 시작 시 재다운로드를 건너뛴다. 볼륨을 초기화하려면:

```bash
docker volume rm ros2-ai-planner_graspgen_ext
```

**cuRobo YAML 설정 주의:** `src/team_8/config/ur5_curobo.yml`은 ASCII 전용으로 유지해야 한다. cuRobo의 `load_yaml`이 ASCII 코덱으로 열기 때문에 비ASCII 문자가 있으면 크래시된다.

### Dockerfile 레이어 순서

무거운 레이어(ROS2 apt 패키지, PyTorch CUDA 휠, GraspGenX/cuRobo 소스 설치, 모델 다운로드)가 상단에, `COPY src`가 마지막에 위치해 노드 코드 변경 시 최종 레이어만 무효화된다.

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
# TSDF 맵 수동 리셋
ros2 topic pub --once /curobo/reset_map std_msgs/msg/Empty "{}"
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
ros2 topic echo /curobo/ready --once   # 플래너 초기화 확인
```

### 환경 및 모델 파일 확인

```bash
docker compose run --rm ai_planner env | grep -E 'ROS_DOMAIN_ID|ROS_LOCALHOST_ONLY|RMW_IMPLEMENTATION|FASTDDS_BUILTIN_TRANSPORTS|GEMINI_API_KEY'
```

```bash
ls /opt/GraspGenX
ls /opt/GraspGenX/ext          # graspgen_ext 볼륨 마운트 포인트
ls /opt/GraspGenModels/checkpoints
ls /opt/models/sam2
```

### 아티팩트 경로

```text
./artifacts/segmentation_service/
./artifacts/graspgen_service/
```
