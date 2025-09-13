import docker
import os
import tempfile
import shutil
from typing import Dict, List, Any

class CodeExecutor:
    """
    A sandboxed code executor using Docker.

    This class runs arbitrary Python code in a secure, isolated Docker
    container to prevent any harm to the host system. It captures and
    returns the output, errors, and any files created by the code.
    """
    def __init__(self, python_image: str = "python:3.11-slim"):
        """
        Initializes the CodeExecutor.

        Args:
            python_image: The Docker image to use for execution.

        Raises:
            Exception: If the Docker client cannot be initialized.
        """
        try:
            self.client = docker.from_env()
            # Check if Docker is running
            self.client.ping()
        except Exception as e:
            print(f"Error initializing Docker client: {e}")
            print("Please ensure Docker is running.")
            raise

        self.python_image = python_image

    def execute(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Executes a string of Python code in a sandboxed environment.

        Args:
            code: The Python code to execute.
            timeout: The maximum execution time in seconds.

        Returns:
            A dictionary containing:
            - 'success': A boolean indicating if the code ran without errors.
            - 'stdout': The standard output from the code.
            - 'stderr': The standard error from the code.
            - 'artifacts': A list of filenames created by the code.
        """
        # Create a temporary directory on the host
        host_temp_dir = tempfile.mkdtemp()

        # Path for the script inside the container
        container_script_path = "/app/script.py"

        # Write the code to a file in the host's temp directory
        with open(os.path.join(host_temp_dir, "script.py"), "w") as f:
            f.write(code)

        container = None
        try:
            container = self.client.containers.run(
                self.python_image,
                command=["python", container_script_path],
                volumes={host_temp_dir: {'bind': '/app', 'mode': 'rw'}},
                working_dir="/app",
                detach=True,
                remove=False # We will remove it manually after getting logs
            )

            # Wait for the container to finish, with a timeout
            result = container.wait(timeout=timeout)

            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')

            # Check execution status
            success = (result['StatusCode'] == 0)

            # List artifacts created in the temp directory
            artifacts = [f for f in os.listdir(host_temp_dir) if f != 'script.py']

        except docker.errors.ContainerError as e:
            success = False
            stdout = ""
            stderr = str(e)
            artifacts = []
        except Exception as e:
            # This can catch timeouts from container.wait()
            success = False
            stdout = ""
            stderr = f"Execution failed: {str(e)}"
            artifacts = []
        finally:
            # Ensure container is stopped and removed
            if container:
                try:
                    container.stop()
                except docker.errors.APIError:
                    pass # container might be already stopped
                try:
                    container.remove()
                except docker.errors.APIError:
                    pass # container might be already removed
            # Clean up the temporary directory
            shutil.rmtree(host_temp_dir)

        return {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'artifacts': artifacts
        }
