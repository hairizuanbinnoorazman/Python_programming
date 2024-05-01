# Code Execution via Docker

This is a simple website that allow us to provide code that would be executed.  
However, it is important to take note that running third-party code on a server is usually a bad idea - we need to isolate the impact of such third party code as much as possible.

## Quickstart

```bash
flask run --host=0.0.0.0
```


## Important Configurations

Some of the configurations we need to add to reduce impact:

- Resource limits
  - CPU Limits
  - Memory Limits
- No mounting of any critical folders into the container

However, we would still need to pass the code into the container - we can simply mount the 