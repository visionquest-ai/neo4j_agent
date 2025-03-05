import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Generator
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import copy
import json
import time
from neo4j import GraphDatabase

# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício Ceolin

START = "__start__"
END = "__end__"

class StateGraph:
    """
    A graph-based state machine for managing complex workflows using Neo4j.

    This class allows defining states, transitions, and conditions for state changes,
    as well as executing the workflow based on the defined graph structure.

    Attributes:
        state_schema (Dict[str, Any]): The schema defining the structure of the state.
        uri (str): Neo4j connection URI.
        user (str): Neo4j username.
        password (str): Neo4j password.
        database (str): Neo4j database name.
        interrupt_before (List[str]): Nodes to interrupt before execution.
        interrupt_after (List[str]): Nodes to interrupt after execution.
    """

    def __init__(self, state_schema: Dict[str, Any], uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "neo4j", database: str = "neo4j",
                 raise_exceptions: bool = False):
        """
        Initialize the StateGraph with Neo4j connection.

        Args:
            state_schema (Dict[str, Any]): The schema defining the structure of the state.
            uri (str): Neo4j connection URI.
            user (str): Neo4j username.
            password (str): Neo4j password.
            database (str): Neo4j database name.
            raise_exceptions (bool): If True, exceptions in node functions will be raised instead of being handled internally.
        """
        self.state_schema = state_schema
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.interrupt_before: List[str] = []
        self.interrupt_after: List[str] = []
        self.raise_exceptions = raise_exceptions
        self.parallel_sync = {}

        # Store functions in memory since Neo4j can't store callables
        self._node_functions = {}
        self._edge_conditions = {}
        self._condition_maps = {}

        # For serialization of conditional values in Neo4j
        self._condition_value_map = {}

        # Initialize Neo4j connection and setup graph
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._setup_graph()

    def _setup_graph(self):
        """
        Set up the Neo4j graph by creating constraints and initial nodes.
        """
        with self._driver.session(database=self.database) as session:
            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.name IS UNIQUE")

            # Add START and END nodes
            session.run(
                "MERGE (n:Node {name: $name}) SET n.type = 'system'",
                name=START
            )
            session.run(
                "MERGE (n:Node {name: $name}) SET n.type = 'system'",
                name=END
            )

    def __del__(self):
        """
        Close Neo4j driver when object is destroyed.
        """
        if hasattr(self, '_driver'):
            self._driver.close()

    def add_node(self, node: str, run: Optional[Callable[..., Any]] = None) -> None:
        """
        Add a node to the graph.

        Args:
            node (str): The name of the node.
            run (Optional[Callable[..., Any]]): The function to run when this node is active.

        Raises:
            ValueError: If the node already exists in the graph.
        """
        with self._driver.session(database=self.database) as session:
            # Check if node exists
            result = session.run("MATCH (n:Node {name: $name}) RETURN n", name=node)
            if result.single():
                raise ValueError(f"Node '{node}' already exists in the graph.")

            # Create node
            session.run("CREATE (n:Node {name: $name, type: 'user'})", name=node)

            # Store function in memory dictionary
            if run:
                self._node_functions[node] = run

    def add_edge(self, in_node: str, out_node: str) -> None:
        """
        Add an unconditional edge between two nodes.

        Args:
            in_node (str): The source node.
            out_node (str): The target node.

        Raises:
            ValueError: If either node doesn't exist in the graph.
        """
        with self._driver.session(database=self.database) as session:
            # Check if nodes exist
            result = session.run(
                "MATCH (n1:Node {name: $in_node}) "
                "MATCH (n2:Node {name: $out_node}) "
                "RETURN n1, n2",
                in_node=in_node, out_node=out_node
            )

            if not result.single():
                raise ValueError("Both nodes must exist in the graph.")

            # Create edge
            session.run(
                "MATCH (n1:Node {name: $in_node}) "
                "MATCH (n2:Node {name: $out_node}) "
                "CREATE (n1)-[r:TRANSITION {unconditional: true}]->(n2)",
                in_node=in_node, out_node=out_node
            )

            # Use a constant true function for unconditional edges
            edge_id = f"{in_node}_to_{out_node}"
            self._edge_conditions[edge_id] = lambda **kwargs: True
            self._condition_maps[edge_id] = {True: out_node}

    def add_conditional_edges(self, in_node: str, func: Callable[..., Any], cond: Dict[Any, str]) -> None:
        """
        Add conditional edges from a node based on a function's output.

        Args:
            in_node (str): The source node.
            func (Callable[..., Any]): The function to determine the next node.
            cond (Dict[Any, str]): Mapping of function outputs to target nodes.

        Raises:
            ValueError: If the source node doesn't exist or if any target node is invalid.
        """
        with self._driver.session(database=self.database) as session:
            # Check if in_node exists
            result = session.run("MATCH (n:Node {name: $name}) RETURN n", name=in_node)
            if not result.single():
                raise ValueError(f"Node '{in_node}' does not exist in the graph.")

            # Check if target nodes exist and create edges
            for cond_value, out_node in cond.items():
                result = session.run("MATCH (n:Node {name: $name}) RETURN n", name=out_node)
                if not result.single():
                    raise ValueError(f"Target node '{out_node}' does not exist in the graph.")

                # Create edge for this condition
                # Use a JSON string representation of the condition value for the edge property
                # For simplicity, we're treating all condition values as strings in Neo4j
                str_cond_value = str(cond_value)
                session.run(
                    "MATCH (n1:Node {name: $in_node}) "
                    "MATCH (n2:Node {name: $out_node}) "
                    "CREATE (n1)-[r:TRANSITION {conditional: true, value: $value}]->(n2)",
                    in_node=in_node, out_node=out_node, value=str_cond_value
                )

            # Store function and condition map in memory
            edge_id = f"{in_node}_conditional"
            self._edge_conditions[edge_id] = func
            self._condition_maps[edge_id] = cond

    def add_parallel_edge(self, in_node: str, out_node: str, fan_in_node: str) -> None:
        """
        Add an unconditional parallel edge between two nodes.

        Args:
            in_node (str): The source node.
            out_node (str): The target node.
            fan_in_node (str): The fan-in node that this parallel flow will reach.

        Raises:
            ValueError: If either node doesn't exist in the graph.
        """
        with self._driver.session(database=self.database) as session:
            # Check if nodes exist
            result = session.run(
                "MATCH (n1:Node {name: $in_node}) "
                "MATCH (n2:Node {name: $out_node}) "
                "MATCH (n3:Node {name: $fan_in_node}) "
                "RETURN n1, n2, n3",
                in_node=in_node, out_node=out_node, fan_in_node=fan_in_node
            )

            if not result.single():
                raise ValueError("All nodes must exist in the graph.")

            # Create edge
            session.run(
                "MATCH (n1:Node {name: $in_node}) "
                "MATCH (n2:Node {name: $out_node}) "
                "CREATE (n1)-[r:TRANSITION {parallel: true, fan_in_node: $fan_in_node}]->(n2)",
                in_node=in_node, out_node=out_node, fan_in_node=fan_in_node
            )

            # Use a constant true function for parallel edges
            edge_id = f"{in_node}_to_{out_node}_parallel"
            self._edge_conditions[edge_id] = lambda **kwargs: True
            self._condition_maps[edge_id] = {True: out_node}

    def add_fanin_node(self, node: str, run: Optional[Callable[..., Any]] = None) -> None:
        """
        Add a fan-in node to the graph.

        Args:
            node (str): The name of the node.
            run (Optional[Callable[..., Any]]): The function to run when this node is active.

        Raises:
            ValueError: If the node already exists in the graph.
        """
        with self._driver.session(database=self.database) as session:
            # Check if node exists
            result = session.run("MATCH (n:Node {name: $name}) RETURN n", name=node)
            if result.single():
                raise ValueError(f"Node '{node}' already exists in the graph.")

            # Create node with fan_in property
            session.run(
                "CREATE (n:Node {name: $name, type: 'user', fan_in: true})",
                name=node
            )

            # Store function in memory dictionary
            if run:
                self._node_functions[node] = run

    def invoke(self, input_state: Dict[str, Any] = {}, config: Dict[str, Any] = {}) -> Generator[Dict[str, Any], None, None]:
        """
        Execute the graph, yielding interrupts and the final state.

        Args:
            input_state (Dict[str, Any]): The initial state.
            config (Dict[str, Any]): Configuration for the execution.

        Yields:
            Dict[str, Any]: Interrupts and the final state during execution.
        """
        current_node = START
        state = input_state.copy()
        config = config.copy()
        # Create a ThreadPoolExecutor for parallel flows
        executor = ThreadPoolExecutor()
        # Mapping from fan-in nodes to list of futures
        fanin_futures: Dict[str, List[Future]] = {}
        # Lock for thread-safe operations on fanin_futures
        fanin_lock = threading.Lock()

        try:
            while current_node != END:
                # Check for interrupt before
                if current_node in self.interrupt_before:
                    yield {"type": "interrupt", "node": current_node, "state": state.copy()}

                # Get node data and run function
                node_data = self.node(current_node)
                node_key = current_node
                run_func = self._node_functions.get(node_key)

                # Execute node's run function if present
                if run_func:
                    try:
                        result = self._execute_node_function(run_func, state, config, current_node)
                        state.update(result)
                    except Exception as e:
                        if self.raise_exceptions:
                            raise RuntimeError(f"Error in node '{current_node}': {str(e)}") from e
                        else:
                            yield {"type": "error", "node": current_node, "error": str(e), "state": state.copy()}
                            return

                # Check for interrupt after
                if current_node in self.interrupt_after:
                    yield {"type": "interrupt", "node": current_node, "state": state.copy()}

                # Get all outgoing edges
                outgoing_edges = self.get_outgoing_edges(current_node)

                # Separate parallel and normal edges
                parallel_edges = []
                normal_edges = []

                for edge in outgoing_edges:
                    if edge.get('parallel', False):
                        fan_in_node = edge.get('fan_in_node')
                        parallel_edges.append((edge['target'], fan_in_node))
                    else:
                        normal_edges.append(edge)

                # Start parallel flows
                for successor, fan_in_node in parallel_edges:
                    # Start a new thread for the flow starting from successor
                    future = executor.submit(self._execute_flow, successor, copy.deepcopy(state), config.copy(), fan_in_node)
                    # Register the future with the corresponding fan-in node
                    with fanin_lock:
                        if fan_in_node not in fanin_futures:
                            fanin_futures[fan_in_node] = []
                        fanin_futures[fan_in_node].append(future)

                # Handle normal edges
                if normal_edges:
                    # Find the next node based on edge conditions
                    next_node = None
                    for edge in normal_edges:
                        edge_id = f"{current_node}_to_{edge['target']}"
                        if edge.get('conditional', False):
                            edge_id = f"{current_node}_conditional"

                        cond_func = self._edge_conditions.get(edge_id, lambda **kwargs: True)
                        cond_map = self._condition_maps.get(edge_id, {True: edge['target']})

                        available_params = {"state": state, "config": config, "node": current_node, "graph": self}
                        cond_params = self._prepare_function_params(cond_func, available_params)
                        cond_result = cond_func(**cond_params)

                        next_node_candidate = cond_map.get(cond_result)
                        if next_node_candidate:
                            next_node = next_node_candidate
                            break

                    if next_node:
                        current_node = next_node
                    else:
                        error_msg = f"No valid next node found from node '{current_node}'"
                        if self.raise_exceptions:
                            raise RuntimeError(error_msg)
                        else:
                            yield {"type": "error", "node": current_node, "error": error_msg, "state": state.copy()}
                            return
                else:
                    # No normal successors
                    # Check if there is a fan-in node with pending futures
                    if fanin_futures:
                        # Proceed to the fan-in node
                        current_node = list(fanin_futures.keys())[0]
                        # Wait for all futures corresponding to this fan-in node
                        futures = fanin_futures.get(current_node, [])
                        results = [future.result() for future in futures]
                        # Collect the results in the state
                        state['parallel_results'] = results

                        # Execute the fan-in node's run function
                        node_key = current_node
                        run_func = self._node_functions.get(node_key)
                        if run_func:
                            try:
                                result = self._execute_node_function(run_func, state, config, current_node)
                                state.update(result)
                            except Exception as e:
                                if self.raise_exceptions:
                                    raise RuntimeError(f"Error in node '{current_node}': {str(e)}") from e
                                else:
                                    yield {"type": "error", "node": current_node, "error": str(e), "state": state.copy()}
                                    return

                        # Continue to next node using _get_next_node
                        next_node = self._get_next_node(current_node, state, config)
                        if not next_node:
                            error_msg = f"No valid next node found from node '{current_node}'"
                            if self.raise_exceptions:
                                raise RuntimeError(error_msg)
                            else:
                                yield {"type": "error", "node": current_node, "error": error_msg, "state": state.copy()}
                                return
                        current_node = next_node
                    else:
                        error_msg = f"No valid next node found from node '{current_node}'"
                        if self.raise_exceptions:
                            raise RuntimeError(error_msg)
                        else:
                            yield {"type": "error", "node": current_node, "error": error_msg, "state": state.copy()}
                            return
        finally:
            executor.shutdown(wait=True)
        # Once END is reached, yield final state
        yield {"type": "final", "state": state.copy()}

    def _execute_flow(self, current_node, state, config, fan_in_node):
        """
        Execute a flow starting from current_node until it reaches fan_in_node.

        Args:
            current_node (str): The starting node of the flow.
            state (Dict[str, Any]): The state of the flow.
            config (Dict[str, Any]): Configuration for the execution.
            fan_in_node (str): The fan-in node where this flow should stop.

        Returns:
            Dict[str, Any]: The final state of the flow when it reaches the fan-in node.
        """
        while current_node != END:
            # Check if current node is the fan-in node
            if current_node == fan_in_node:
                # Return the state to be collected at fan-in node
                return state

            # Get node run function
            node_key = current_node
            run_func = self._node_functions.get(node_key)

            # Execute node's run function if present
            if run_func:
                try:
                    result = self._execute_node_function(run_func, state, config, current_node)
                    state.update(result)
                except Exception as e:
                    if self.raise_exceptions:
                        raise RuntimeError(f"Error in node '{current_node}': {str(e)}") from e
                    else:
                        # Return error in state
                        state['error'] = str(e)
                        return state

            # Determine next node
            try:
                next_node = self._get_next_node(current_node, state, config)
            except Exception as e:
                if self.raise_exceptions:
                    raise
                else:
                    state['error'] = str(e)
                    return state

            if next_node:
                current_node = next_node
            else:
                error_msg = f"No valid next node found from node '{current_node}' in parallel flow"
                if self.raise_exceptions:
                    raise RuntimeError(error_msg)
                else:
                    state['error'] = error_msg
                    return state

        # Reached END
        return state

    def _execute_node_function(self, func: Callable[..., Any], state: Dict[str, Any], config: Dict[str, Any], node: str) -> Dict[str, Any]:
        """
        Execute the function associated with a node.

        Args:
            func (Callable[..., Any]): The function to execute.
            state (Dict[str, Any]): The current state.
            config (Dict[str, Any]): The configuration.
            node (str): The current node name.

        Returns:
            Dict[str, Any]: The result of the function execution.

        Raises:
            Exception: If an exception occurs during function execution.
        """
        available_params = {"state": state, "config": config, "node": node, "graph": self}
        if 'parallel_results' in state:
            available_params['parallel_results'] = state['parallel_results']
        function_params = self._prepare_function_params(func, available_params)
        result = func(**function_params)
        if isinstance(result, dict):
            return result
        else:
            # If result is not a dict, wrap it in a dict
            return {"result": result}

    def _prepare_function_params(self, func: Callable[..., Any], available_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the parameters for a node function based on its signature.

        Args:
            func (Callable[..., Any]): The function to prepare parameters for.
            available_params (Dict[str, Any]): Dictionary of available parameters.

        Returns:
            Dict[str, Any]: The prepared parameters for the function.

        Raises:
            ValueError: If required parameters for the function are not provided.
        """
        sig = inspect.signature(func)
        function_params = {}

        if len(sig.parameters) == 0:
            return {}

        for param_name, param in sig.parameters.items():
            if param_name in available_params:
                function_params[param_name] = available_params[param_name]
            elif param.default is not inspect.Parameter.empty:
                function_params[param_name] = param.default
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                function_params.update({k: v for k, v in available_params.items() if k not in function_params})
                break
            else:
                raise ValueError(f"Required parameter '{param_name}' not provided for function '{func.__name__}'")

        return function_params

    def set_entry_point(self, init_state: str) -> None:
        """
        Set the entry point of the graph.

        Args:
            init_state (str): The initial state node.

        Raises:
            ValueError: If the initial state node doesn't exist in the graph.
        """
        with self._driver.session(database=self.database) as session:
            # Check if node exists
            result = session.run("MATCH (n:Node {name: $name}) RETURN n", name=init_state)
            if not result.single():
                raise ValueError(f"Node '{init_state}' does not exist in the graph.")

            # Create edge from START to init_state
            session.run(
                "MATCH (start:Node {name: $start}) "
                "MATCH (init:Node {name: $init}) "
                "CREATE (start)-[r:TRANSITION {unconditional: true}]->(init)",
                start=START, init=init_state
            )

            # Add to in-memory condition dictionaries
            edge_id = f"{START}_to_{init_state}"
            self._edge_conditions[edge_id] = lambda **kwargs: True
            self._condition_maps[edge_id] = {True: init_state}

    def set_finish_point(self, final_state: str) -> None:
        """
        Set the finish point of the graph.

        Args:
            final_state (str): The final state node.

        Raises:
            ValueError: If the final state node doesn't exist in the graph.
        """
        with self._driver.session(database=self.database) as session:
            # Check if node exists
            result = session.run("MATCH (n:Node {name: $name}) RETURN n", name=final_state)
            if not result.single():
                raise ValueError(f"Node '{final_state}' does not exist in the graph.")

            # Create edge from final_state to END
            session.run(
                "MATCH (final:Node {name: $final}) "
                "MATCH (end:Node {name: $end}) "
                "CREATE (final)-[r:TRANSITION {unconditional: true}]->(end)",
                final=final_state, end=END
            )

            # Add to in-memory condition dictionaries
            edge_id = f"{final_state}_to_{END}"
            self._edge_conditions[edge_id] = lambda **kwargs: True
            self._condition_maps[edge_id] = {True: END}

    def compile(self, interrupt_before: List[str] = [], interrupt_after: List[str] = []) -> 'StateGraph':
        """
        Compile the graph and set interruption points.

        Args:
            interrupt_before (List[str]): Nodes to interrupt before execution.
            interrupt_after (List[str]): Nodes to interrupt after execution.

        Returns:
            StateGraph: The compiled graph instance.

        Raises:
            ValueError: If any interrupt node doesn't exist in the graph.
        """
        with self._driver.session(database=self.database) as session:
            for node in interrupt_before + interrupt_after:
                result = session.run("MATCH (n:Node {name: $name}) RETURN n", name=node)
                if not result.single():
                    raise ValueError(f"Interrupt node '{node}' does not exist in the graph.")

        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after
        return self

    def node(self, node_name: str) -> Dict[str, Any]:
        """
        Get the attributes of a specific node.

        Args:
            node_name (str): The name of the node.

        Returns:
            Dict[str, Any]: The node's attributes.

        Raises:
            KeyError: If the node is not found in the graph.
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n:Node {name: $name}) RETURN properties(n) as props",
                name=node_name
            )
            record = result.single()
            if not record:
                raise KeyError(f"Node '{node_name}' not found in the graph")

            # Get props from Neo4j
            props = dict(record["props"])

            # Add the run function from memory if it exists
            run_func = self._node_functions.get(node_name)
            if run_func:
                props["run"] = run_func

            return props

    def edge(self, in_node: str, out_node: str) -> Dict[str, Any]:
        """
        Get the attributes of a specific edge.

        Args:
            in_node (str): The source node.
            out_node (str): The target node.

        Returns:
            Dict[str, Any]: The edge's attributes.

        Raises:
            KeyError: If the edge is not found in the graph.
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n1:Node {name: $in_node})-[r:TRANSITION]->(n2:Node {name: $out_node}) "
                "RETURN properties(r) as props",
                in_node=in_node, out_node=out_node
            )
            record = result.single()
            if not record:
                raise KeyError(f"Edge from '{in_node}' to '{out_node}' not found in the graph")

            # Get props from Neo4j
            props = dict(record["props"])
            props["source"] = in_node
            props["target"] = out_node

            # Add condition function from memory
            edge_id = f"{in_node}_to_{out_node}"
            if props.get('conditional', False):
                edge_id = f"{in_node}_conditional"

            cond_func = self._edge_conditions.get(edge_id)
            cond_map = self._condition_maps.get(edge_id)

            if cond_func:
                props["cond"] = cond_func
            if cond_map:
                props["cond_map"] = cond_map

            return props

    def get_outgoing_edges(self, node: str) -> List[Dict[str, Any]]:
        """
        Get all outgoing edges for a node.

        Args:
            node (str): The source node.

        Returns:
            List[Dict[str, Any]]: List of edge dictionaries with their properties.

        Raises:
            KeyError: If the node is not found in the graph.
        """
        with self._driver.session(database=self.database) as session:
            # First check if node exists
            node_result = session.run("MATCH (n:Node {name: $name}) RETURN n", name=node)
            if not node_result.single():
                raise KeyError(f"Node '{node}' not found in the graph")

            # Get all outgoing edges
            result = session.run(
                "MATCH (n1:Node {name: $node})-[r:TRANSITION]->(n2:Node) "
                "RETURN n2.name as target, properties(r) as props",
                node=node
            )

            edges = []
            for record in result:
                props = dict(record["props"])
                props["source"] = node
                props["target"] = record["target"]

                # Add function references
                edge_id = f"{node}_to_{props['target']}"
                if props.get('conditional', False):
                    edge_id = f"{node}_conditional"
                elif props.get('parallel', False):
                    edge_id = f"{node}_to_{props['target']}_parallel"

                cond_func = self._edge_conditions.get(edge_id)
                cond_map = self._condition_maps.get(edge_id)

                if cond_func:
                    props["cond"] = cond_func
                if cond_map:
                    props["cond_map"] = cond_map

                edges.append(props)

            return edges

    def successors(self, node: str) -> List[str]:
        """
        Get the list of successors for a given node.

        Args:
            node (str): The name of the node.

        Returns:
            List[str]: A list of successor node names.

        Raises:
            KeyError: If the node is not found in the graph.
        """
        with self._driver.session(database=self.database) as session:
            # First check if node exists
            node_result = session.run("MATCH (n:Node {name: $name}) RETURN n", name=node)
            if not node_result.single():
                raise KeyError(f"Node '{node}' not found in the graph")

            # Get successors
            result = session.run(
                "MATCH (n1:Node {name: $node})-[r:TRANSITION]->(n2:Node) "
                "RETURN n2.name as successor",
                node=node
            )

            return [record["successor"] for record in result]

    def stream(self, input_state: Dict[str, Any] = {}, config: Dict[str, Any] = {}) -> Generator[Dict[str, Any], None, None]:
        """
        Execute the graph, yielding results at each node execution, including interrupts.

        Args:
            input_state (Dict[str, Any]): The initial state.
            config (Dict[str, Any]): Configuration for the execution.

        Yields:
            Dict[str, Any]: Intermediate states, interrupts, errors, and the final state during execution.
        """
        current_node = START
        state = input_state.copy()
        config = config.copy()

        while current_node != END:
            # Check for interrupt before
            if current_node in self.interrupt_before:
                yield {"type": "interrupt_before", "node": current_node, "state": state.copy()}

            # Get node run function
            node_key = current_node
            run_func = self._node_functions.get(node_key)

            # Execute node's run function if present
            if run_func:
                try:
                    result = self._execute_node_function(run_func, state, config, current_node)
                    state.update(result)
                    # Yield intermediate state after execution
                    yield {"type": "state", "node": current_node, "state": state.copy()}
                except Exception as e:
                    if self.raise_exceptions:
                        raise RuntimeError(f"Error in node '{current_node}': {str(e)}") from e
                    else:
                        yield {"type": "error", "node": current_node, "error": str(e), "state": state.copy()}
                        return

            # Check for interrupt after
            if current_node in self.interrupt_after:
                yield {"type": "interrupt_after", "node": current_node, "state": state.copy()}

            # Determine next node
            next_node = self._get_next_node(current_node, state, config)
            if not next_node:
                error_msg = f"No valid next node found from node '{current_node}'"
                if self.raise_exceptions:
                    raise RuntimeError(error_msg)
                else:
                    yield {"type": "error", "node": current_node, "error": error_msg, "state": state.copy()}
                    return

            current_node = next_node

        # Once END is reached, yield final state
        yield {"type": "final", "state": state.copy()}

    def _get_next_node(self, current_node: str, state: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
        """
        Determine the next node based on the current node's successors and conditions.

        Args:
            current_node (str): The current node.
            state (Dict[str, Any]): The current state.
            config (Dict[str, Any]): The configuration.

        Returns:
            Optional[str]: The name of the next node, or None if no valid next node is found.
        """
        outgoing_edges = self.get_outgoing_edges(current_node)

        for edge in outgoing_edges:
            if edge.get('parallel', False):
                # Skip parallel edges for direct traversal
                continue

            cond_func = edge.get("cond", lambda **kwargs: True)
            cond_map = edge.get("cond_map", None)
            available_params = {"state": state, "config": config, "node": current_node, "graph": self}
            cond_params = self._prepare_function_params(cond_func, available_params)
            cond_result = cond_func(**cond_params)

            if cond_map:
                # cond_map is a mapping from condition results to nodes
                next_node = cond_map.get(cond_result, None)
                if next_node:
                    return next_node
            else:
                # cond_result is treated as boolean
                if cond_result:
                    return edge['target']

        # No valid next node found
        return None

    def _prepare_function_params(self, func: Callable[..., Any], available_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the parameters for a node function based on its signature.

        Args:
            func (Callable[..., Any]): The function to prepare parameters for.
            available_params (Dict[str, Any]): Dictionary of available parameters.

        Returns:
            Dict[str, Any]: The prepared parameters for the function.

        Raises:
            ValueError: If required parameters for the function are not provided.
        """
        sig = inspect.signature(func)
        function_params = {}

        if len(sig.parameters) == 0:
            return {}

        for param_name, param in sig.parameters.items():
            if param_name in available_params:
                function_params[param_name] = available_params[param_name]
            elif param.default is not inspect.Parameter.empty:
                function_params[param_name] = param.default
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                function_params.update({k: v for k, v in available_params.items() if k not in function_params})
                break
            else:
                raise ValueError(f"Required parameter '{param_name}' not provided for function '{func.__name__}'")

        return function_params

    def _execute_node_function(self, func: Callable[..., Any], state: Dict[str, Any], config: Dict[str, Any], node: str) -> Dict[str, Any]:
        """
        Execute the function associated with a node.

        Args:
            func (Callable[..., Any]): The function to execute.
            state (Dict[str, Any]): The current state.
            config (Dict[str, Any]): The configuration.
            node (str): The current node name.

        Returns:
            Dict[str, Any]: The result of the function execution.

        Raises:
            Exception: If an exception occurs during function execution.
        """
        available_params = {"state": state, "config": config, "node": node, "graph": self}
        if 'parallel_results' in state:
            available_params['parallel_results'] = state['parallel_results']
        function_params = self._prepare_function_params(func, available_params)
        result = func(**function_params)
        if isinstance(result, dict):
            return result
        else:
            # If result is not a dict, wrap it in a dict
            return {"result": result}

    def _setup_graph(self):
        """
        Set up the Neo4j graph by creating constraints and initial nodes.
        """
        with self._driver.session(database=self.database) as session:
            try:
                # Create constraints
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.name IS UNIQUE")
            except Exception as e:
                print(f"Warning: Could not create constraint: {e}")

            # Ensure the START and END nodes exist
            session.run(
                """
                MERGE (start:Node {name: $start_name})
                SET start.type = 'system'
                MERGE (end:Node {name: $end_name})
                SET end.type = 'system'
                """,
                start_name=START, end_name=END
            )

            # Verify the nodes were created
            result = session.run(
                """
                MATCH (n:Node)
                WHERE n.name IN [$start_name, $end_name]
                RETURN n.name as name
                """,
                start_name=START, end_name=END
            )

            found_nodes = [record["name"] for record in result]
            if START not in found_nodes or END not in found_nodes:
                print(f"Warning: Failed to create START/END nodes properly. Found: {found_nodes}")

    def node_exists(self, node_name: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node_name (str): The name of the node.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n:Node {name: $name}) RETURN count(n) as count",
                name=node_name
            )
            record = result.single()
            return record and record["count"] > 0

    def node(self, node_name: str) -> Dict[str, Any]:
        """
        Get the attributes of a specific node.

        Args:
            node_name (str): The name of the node.

        Returns:
            Dict[str, Any]: The node's attributes.

        Raises:
            KeyError: If the node is not found in the graph.
        """
        with self._driver.session(database=self.database) as session:
            # Check if we're looking for START or END and they don't exist
            if node_name in [START, END] and not self.node_exists(node_name):
                print(f"Re-creating missing system node: {node_name}")
                # Re-create the system node
                session.run(
                    "MERGE (n:Node {name: $name}) SET n.type = 'system'",
                    name=node_name
                )

            result = session.run(
                "MATCH (n:Node {name: $name}) RETURN properties(n) as props",
                name=node_name
            )
            record = result.single()
            if not record:
                raise KeyError(f"Node '{node_name}' not found in the graph")

            # Get props from Neo4j
            props = dict(record["props"])
            props["name"] = node_name

            # Add the run function from memory if it exists
            run_func = self._node_functions.get(node_name)
            if run_func:
                props["run"] = run_func

            return props

    def clear_graph(self):
        """
        Clear all nodes and edges from the graph database.
        Use with caution!
        """
        with self._driver.session(database=self.database) as session:
            # Delete all nodes and relationships in the database
            session.run("MATCH (n:Node) DETACH DELETE n")

            # Clear in-memory function storage
            self._node_functions = {}
            self._edge_conditions = {}
            self._condition_maps = {}

            # Re-initialize with START and END nodes
            self._setup_graph()

            # Verify START and END nodes were created
            if not self.node_exists(START) or not self.node_exists(END):
                print(f"Warning: START or END nodes not created properly after clear_graph!")
                print(f"  START exists: {self.node_exists(START)}")
                print(f"  END exists: {self.node_exists(END)}")


    def export_to_json(self, filename="state_graph.json"):
       """
       Export the graph structure to a JSON file.

       Args:
           filename (str): The name of the file to save the JSON data to.
       """
       import json

       with self._driver.session(database=self.database) as session:
           # Get all nodes
           node_result = session.run(
               "MATCH (n:Node) RETURN n.name as name, properties(n) as props"
           )
           nodes = []
           for record in node_result:
               node = dict(record["props"])
               node["name"] = record["name"]
               # Remove Neo4j specific properties
               if "run" in node:
                   del node["run"]
               nodes.append(node)

           # Get all edges
           edge_result = session.run(
               "MATCH (n1:Node)-[r:TRANSITION]->(n2:Node) "
               "RETURN n1.name as source, n2.name as target, properties(r) as props"
           )
           edges = []
           for record in edge_result:
               edge = dict(record["props"])
               edge["source"] = record["source"]
               edge["target"] = record["target"]
               edges.append(edge)

           # Combine into a complete graph representation
           graph_data = {
               "nodes": nodes,
               "edges": edges,
               "interrupt_before": self.interrupt_before,
               "interrupt_after": self.interrupt_after
           }

           # Write to file
           with open(filename, 'w') as f:
               json.dump(graph_data, f, indent=2)

    def visualize_graph(self):
        """
        Generate Cypher query to visualize the graph in Neo4j Browser.

        Returns:
            str: Cypher query for visualization.
        """
        query = """
        MATCH p=(n:Node)-[r:TRANSITION]->(m:Node)
        RETURN p
        LIMIT 100
        """
        return query

    def render_graphviz(self, output_file="state_graph.dot"):
        """
        Export the graph structure to a GraphViz DOT file format.

        This method queries the Neo4j database and generates a DOT file
        that can be used with GraphViz tools for visualization.

        Args:
            output_file (str): The name of the DOT file to create.

        Returns:
            str: Path to the created DOT file.
        """
        with self._driver.session(database=self.database) as session:
            # Get all nodes
            node_result = session.run(
                "MATCH (n:Node) RETURN n.name as name, properties(n) as props"
            )

            # Get all edges
            edge_result = session.run(
                "MATCH (n1:Node)-[r:TRANSITION]->(n2:Node) "
                "RETURN n1.name as source, n2.name as target, properties(r) as props"
            )

            # Create DOT file content
            dot_content = ["digraph StateGraph {"]
            dot_content.append("  rankdir=TB;")
            dot_content.append("  node [shape=rectangle, style=filled, fillcolor=white];")

            # Add nodes
            for record in node_result:
                node_name = record["name"]
                props = record["props"]

                # Create label with relevant properties
                label = f"{node_name}\\n"
                if node_name in self.interrupt_before:
                    label += "interrupt_before: true\\n"
                if node_name in self.interrupt_after:
                    label += "interrupt_after: true\\n"
                if props.get("fan_in", False):
                    label += "fan_in: true\\n"

                # Add special styling for system nodes
                if props.get("type") == "system":
                    if node_name == START:
                        dot_content.append(f'  "{node_name}" [label="{label}", fillcolor=lightgreen];')
                    elif node_name == END:
                        dot_content.append(f'  "{node_name}" [label="{label}", fillcolor=lightpink];')
                    else:
                        dot_content.append(f'  "{node_name}" [label="{label}", fillcolor=lightgrey];')
                else:
                    dot_content.append(f'  "{node_name}" [label="{label}"];')

            # Add edges
            for record in edge_result:
                source = record["source"]
                target = record["target"]
                props = record["props"]

                # Create edge attributes
                attrs = []

                if props.get("parallel", False):
                    attrs.append("color=blue")
                    attrs.append("style=dashed")
                    fan_in = props.get("fan_in_node", "")
                    if fan_in:
                        attrs.append(f'label="parallel to {fan_in}"')

                if props.get("conditional", False):
                    attrs.append("color=red")

                if attrs:
                    attr_str = " [" + ", ".join(attrs) + "]"
                    dot_content.append(f'  "{source}" -> "{target}"{attr_str};')
                else:
                    dot_content.append(f'  "{source}" -> "{target}";')

            dot_content.append("}")

            # Write to file
            with open(output_file, 'w') as f:
                f.write("\n".join(dot_content))

            return output_file

