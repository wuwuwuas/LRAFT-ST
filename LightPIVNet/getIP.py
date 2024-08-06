import json
import os
import platform
import socket


def resolve_master_node(host: str, port: int = 8888):
    slurm_cluster_name = get_environment_variable(identifier="SLURM_CLUSTER_NAME")
    slurm_job_id = get_environment_variable(identifier="SLURM_JOBID")
    slurm_job_name = get_environment_variable(identifier="SLURM_JOB_NAME")
    slurm_node_list = get_environment_variable(identifier="SLURM_NODELIST")
    slurm_number_nodes = get_environment_variable(identifier="SLURM_NNODES")
    slurm_number_tasks = get_environment_variable(identifier="SLURM_NTASKS")
    slurm_tasks_per_node = get_environment_variable(identifier="SLURM_TASKS_PER_NODE")
    
    node_list = create_tf_config(
        node_list=slurm_node_list,
        num_nodes=slurm_number_nodes,
        num_tasks=slurm_number_tasks,
        tasks_per_node=slurm_tasks_per_node,
        port=port
    )
    
    myHostname = socket.gethostname()
    searchString = myHostname + ':8888'
    myRank = node_list.index(searchString)
    myIP = socket.gethostbyname(myHostname)
    
    master_node = node_list[0]
    master_name, master_port = master_node.split(':')
    master_ip = socket.gethostbyname(master_name)
    return master_ip, master_name, master_port, node_list, myHostname, myRank, myIP


def create_tf_config(node_list, num_nodes, num_tasks, tasks_per_node, port=8888):
    # parse integers
    parsed_num_nodes = int(num_nodes)
    parsed_num_tasks = int(num_tasks)
    if "(" in tasks_per_node:
        parsed_tasks_per_node = int(tasks_per_node.split("(")[0])
    else:
        parsed_tasks_per_node = int(tasks_per_node)

    # parse node_list
    if "[" in node_list and "]" in node_list:
        _bracket_index = node_list.find("[")
        _node_prefix = node_list[:_bracket_index]
        _node_numbers = node_list[_bracket_index:].replace("[", "").replace("]", "")
        _nodes = _node_numbers.split(",")
        parsed_list = []
        index = None

        for _node in _nodes:
            if "-" in _node:
                _start_node, _stop_node = _node.split("-")
                _fill_length = len(_start_node)
                _start_node, _stop_node = int(_start_node), int(_stop_node)
                for _node_number in range(_start_node, _stop_node + 1):
                    _host = f"{_node_prefix}{str(_node_number).zfill(_fill_length)}"
                    parsed_list.append(f"{_host}:{port}")
                    if _host == platform.node():
                        index = len(parsed_list) - 1
            else:
                _host = f"{_node_prefix}{_node}"
                parsed_list.append(f"{_host}:{port}")
                if _host == platform.node():
                    index = len(parsed_list) - 1

    else:
        parsed_list = [f"{node_list}:{port}"]
        index = 0

    assert len(parsed_list) == parsed_num_nodes
    assert parsed_num_tasks == parsed_tasks_per_node * parsed_num_nodes
    assert index is not None

    return parsed_list
	
def get_environment_variable(identifier: str) -> str:
    if not isinstance(identifier, str):
        raise TypeError(f"identifier must be of type str but found type(identifier)={type(identifier)} instead.")
    if identifier == "":
        raise ValueError(f"identifier must be non-empty string, yet received empty string.")
    env_variable = os.environ.get(identifier)
    if env_variable is None:
        raise EnvironmentError(f"environment variable ${identifier} is not set")
    return env_variable

#masterIP, masterNode, masterPort,nodeList,myHostname,myRank,myIP = resolve_master_node(platform.node(),8888)
#print('Hostname: ', myHostname, ' Rank: ', myRank, ' myIP: ', myIP, ' MasterNode :', masterNode,' Master IP: ', masterIP, ' MasterPort: ', masterPort)
#print('NodeList: ', nodeList)