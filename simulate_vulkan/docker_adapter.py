# TODO: create makefile
import docker
import os

from docker.models.containers import Container

from pytorch.constants import LEADER_ROLE, WORKER_ROLE, USER_SCRIPT_FILE, LEADER_DIR, WORKER_DIR
from pytorch.exceptions import InvalidRole

from simulate_vulkan.constants import LOCAL_SPLIT_DATA_PATH, LOCAL_USER_SCRIPT_PATH


_client = docker.from_env()


def delete_file_in_container(container: Container, file_path):
    container.exec_run(f"rm -f {file_path}")


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
    print(f"image {image_full_name} created")
    return image


def create_container(role: str, node: int | None = None, worker_nodes_num: int = 0):
    image_full_name = get_image_full_name(role)

    try:
        _client.images.get(image_full_name)
    except docker.errors.ImageNotFound:
        create_image(image_full_name, worker_nodes_num=worker_nodes_num)

    current_dir = os.path.abspath('.')
    parent_dir = os.path.dirname(current_dir)
    volume_host_path = f"{parent_dir}/io/{role.lower()}"
    if node is not None:
        volume_host_path += f"/{node}"
    os.makedirs(volume_host_path, exist_ok=True)

    volumes = {volume_host_path: {"bind": f"/io/{role.lower()}", "mode": "rw"}}

    container = _client.containers.run(
        image_full_name,
        detach=True,
        volumes=volumes
    )

    print(f"container {container.name} created")
    print("volumes:", volumes)

    return container


if __name__ == "__main__":
    import shutil

    leader = create_container(LEADER_ROLE, worker_nodes_num=2)
    worker_0 = create_container(WORKER_ROLE, node=0)
    worker_1 = create_container(WORKER_ROLE, node=1)

    shutil.copy(LOCAL_USER_SCRIPT_PATH, LEADER_DIR / USER_SCRIPT_FILE)
    shutil.copytree(LOCAL_SPLIT_DATA_PATH / "0", WORKER_DIR / "0" / "data")
    shutil.copy(LOCAL_USER_SCRIPT_PATH, WORKER_DIR / "0" / USER_SCRIPT_FILE)
    shutil.copytree(LOCAL_SPLIT_DATA_PATH / "1", WORKER_DIR / "1" / "data")
    shutil.copy(LOCAL_USER_SCRIPT_PATH, WORKER_DIR / "1" / USER_SCRIPT_FILE)
