import docker
import os

from pytorch.constants import LEADER_ROLE, WORKER_ROLE, USER_SCRIPT_FILE, LEADER_DIR, WORKER_DIR
from pytorch.exceptions import InvalidRole

from simulate_vulkan.constants import SPLIT_DATA_PATH


_client = docker.from_env()


def get_image_full_name(role: str):
    if role.upper() not in [LEADER_ROLE, WORKER_ROLE]:
        raise InvalidRole

    return f"{role.lower()}:latest"


def create_image(image_full_name: str, worker_nodes_num: int = 0):
    build_context = "../"
    dockerfile_path = "Dockerfile_pytorch"
    dockerfile_args = {
        "ROLE": image_full_name.split(":")[0].upper(),
        "WORKER_NODES_NUM": str(worker_nodes_num)
    }

    image, build_logs = _client.images.build(
        path=build_context,
        dockerfile=dockerfile_path,
        tag=image_full_name,
        buildargs=dockerfile_args,
        rm=True  # Remove intermediate containers after a successful build
    )

    for log in build_logs:
        print(log)

    return image


def create_container(role: str, node: int | None = None, worker_nodes_num: int = 0):
    image_full_name = get_image_full_name(role)

    try:
        _client.images.get(image_full_name)
    except docker.errors.ImageNotFound:
        create_image(role, worker_nodes_num=worker_nodes_num)

    current_dir = os.path.abspath('.')
    parent_dir = os.path.dirname(current_dir)
    volume_host_path = f"{parent_dir}/io/{role.lower()}"
    if node is not None:
        volume_host_path += f"/{node}"
    os.makedirs(volume_host_path, exist_ok=True)

    volumes = {volume_host_path: {"bind": f"/io/{role.lower()}", "mode": "rw"}}
    print(volumes)

    container = _client.containers.run(
        image_full_name,
        detach=True,
        volumes=volumes
    )
    print(f"{container.id} created")

    return container


if __name__ == "__main__":
    import shutil

    leader = create_container("leader", worker_nodes_num=2)
    shutil.copy(USER_SCRIPT_FILE, LEADER_DIR / USER_SCRIPT_FILE)

    worker_0 = create_container("worker", node=0)
    shutil.copytree(SPLIT_DATA_PATH / "0", WORKER_DIR / "0" / "data")
    shutil.copy(USER_SCRIPT_FILE, WORKER_DIR / "0" / USER_SCRIPT_FILE)

    worker_1 = create_container("worker", node=1)
    shutil.copytree(SPLIT_DATA_PATH / "1", WORKER_DIR / "1" / "data")
    shutil.copy(USER_SCRIPT_FILE, WORKER_DIR / "1" / USER_SCRIPT_FILE)
