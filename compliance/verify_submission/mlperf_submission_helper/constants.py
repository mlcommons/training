"""Constants used throughput the library."""
BENCHMARK_NAMES = [
    "resnet", "ssd", "maskrcnn", "transformer", "gnmt", "ncf", "minigo"
]
SUBM_META_PROPS = ["org", "poc_email"]
ENTRY_META_PROPS = [
    "division", "status", "hardware", "framework", "interconnect", "nodes",
    "os", "libraries", "compilers"
    # Not required
    # "power"
    # "notes",
]
NODE_META_PROPS = [
    "num_nodes", "cpu", "num_cores", "num_vcpus", "accelerator",
    "num_accelerators", "sys_mem_size", "sys_storage_type", "sys_storage_size",
    "cpu_accel_interconnect", "network_card", "num_network_cards", "notes"
]
REQUIRED_RESULT_NUM = {
    "resnet": 5,
    "ssd": 5,
    "maskrcnn": 5,
    "transformer": 10,
    "gnmt": 10,
    "ncf": 100,
    "minigo": 20
}

REFERENCE_RESULTS = {
    "resnet": 1.0,
    "ssd": 1.0,
    "maskrcnn": 1.0,
    "transformer": 1.0,
    "gnmt": 1.0,
    "ncf": 1.0,
    "minigo": 1.0
}

RESULT_SUBM_META_COLUMNS = [
    "org",
]
RESULT_ENTRY_META_COLUMNS = [
    "division", "status", "hardware", "framework",
    # not required
    # "power", "notes"
]

DIVISION_COMPLIANCE_CHECK_LEVEL = {"open": 1, "closed": 2}

# check result status
SUCCESS = "success"
FAILURE = "failure"
ERROR = "error"
