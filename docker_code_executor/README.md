# Code Execution via Docker

This is a simple website that allow us to provide code that would be executed.  
However, it is important to take note that running third-party code on a server is usually a bad idea - we need to isolate the impact of such third party code as much as possible.

## Quickstart

```bash
flask run --host=0.0.0.0
```


## Important Configurations

Some of the configurations we need to add to reduce impact:

- Resource limits (Prevent resource exhaustion)
  - CPU Limits
  - Memory Limits
- Configure `ulimits`. Some possible things to check:
  - No of files to be created
  - File size
- No mounting of any critical folders into the container
- Dropping of all linux capabilities. In most cases, third party code shouldn't need to do fancy stuff -e.g. Access to modify file permissions etc
- Ensuring a timeout for each container and removing it once done (to reduce resource usage)
- Ensure that container is started with non-root user

However, we would still need to pass the code into the container - we can simply mount just a small directory with the code files into the container with "read only" permissions. This reduces the need for us to mangle the start up sequence of the container OR attempt to do container builds (e.g. recently in 2024 - there are security issues when building a container; need to allow containers more access just to be able to create it - and this building process will simply create too many throwaway containers - unnecessary administrative work)