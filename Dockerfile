ARG PLANNER_BASE_IMAGE=ros2-ai-planner-base:latest
FROM ${PLANNER_BASE_IMAGE}

# Workspace
WORKDIR /ros2_ws
COPY src/ src/
COPY scripts/ scripts/
COPY segmented_objects/ segmented_objects/
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

COPY scripts/entrypoint.sh /entrypoint.sh
COPY scripts/start_graspgen_server.sh /start_graspgen_server.sh
RUN chmod +x /entrypoint.sh
RUN chmod +x /start_graspgen_server.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
