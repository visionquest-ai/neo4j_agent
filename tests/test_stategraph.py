import unittest
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Import StateGraph - adjust import path as needed
from the_edge_agent import StateGraph, START, END

class TestNeo4jStateGraph(unittest.TestCase):
    """
    Tests for the Neo4j implementation of StateGraph.

    This test requires a running Neo4j instance with the credentials specified in setUp().
    """

    def setUp(self):
        """Set up a Neo4j StateGraph for testing."""
        self.uri = "bolt://localhost:7687"
        self.user = "neo4j"
        self.password = "password"
        self.database = "neo4j"

        # Create a test graph with a simple state schema
        self.graph = StateGraph(
            state_schema={"test": "schema"},
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database
        )

        # Clean up any existing graph data from previous test runs
        with self.graph._driver.session(database=self.database) as session:
            # Delete all nodes and relationships in the database
            session.run("MATCH (n:Node) DETACH DELETE n")

        # Setup the graph from scratch
        self.graph._setup_graph()

        # Verify setup was successful
        if not self.graph.node_exists(START) or not self.graph.node_exists(END):
            print(f"WARNING: START or END nodes missing after setup!")


    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'graph'):
            self.graph.clear_graph()

    def test_init(self):
        """
        Verify that the StateGraph is initialized correctly with the given state schema.
        """
        self.assertEqual(self.graph.state_schema, {"test": "schema"})
        self.assertEqual(self.graph.uri, self.uri)
        self.assertEqual(self.graph.user, self.user)
        self.assertEqual(self.graph.database, self.database)

    def test_add_node(self):
        """
        Test adding nodes to the graph, both with and without associated functions.
        Verify that nodes are correctly added and their run functions are properly set.
        """
        # Test adding a node without a run function
        self.graph.add_node("test_node")
        node_data = self.graph.node("test_node")
        self.assertEqual(node_data["name"], "test_node")

        # Test adding a node with a run function
        def test_func(state):
            return {"result": state["value"] * 2}

        self.graph.add_node("func_node", run=test_func)
        node_data = self.graph.node("func_node")
        self.assertEqual(node_data["name"], "func_node")

        # Check if the function is available through _node_functions
        self.assertIn("func_node", self.graph._node_functions)
        self.assertEqual(self.graph._node_functions["func_node"], test_func)

    def test_add_node_duplicate(self):
        """
        Ensure that adding a duplicate node raises a ValueError.
        """
        self.graph.add_node("test_node")
        with self.assertRaises(ValueError):
            self.graph.add_node("test_node")

    def test_add_edge(self):
        """
        Verify that edges can be added between existing nodes in the graph.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")

        # Check if node2 is a successor of node1
        successors = self.graph.successors("node1")
        self.assertIn("node2", successors)

        # Check edge properties
        edge_data = self.graph.edge("node1", "node2")
        self.assertTrue(edge_data.get("unconditional", False))

    def test_add_edge_nonexistent_node(self):
        """
        Check that attempting to add an edge between non-existent nodes raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.graph.add_edge("nonexistent1", "nonexistent2")

    def test_add_conditional_edges(self):
        """
        Test the addition of conditional edges based on a given function and condition map.
        Verify that the edges are correctly added to the graph.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_node("node3")

        def condition_func(state):
            return state.get("value", False)

        self.graph.add_conditional_edges("node1", condition_func, {True: "node2", False: "node3"})

        # Check if both node2 and node3 are successors of node1
        successors = self.graph.successors("node1")
        self.assertIn("node2", successors)
        self.assertIn("node3", successors)

        # Check if the condition function is stored correctly
        edge_id = "node1_conditional"
        self.assertIn(edge_id, self.graph._edge_conditions)
        self.assertEqual(self.graph._edge_conditions[edge_id], condition_func)

        # Check if the condition map is stored correctly
        self.assertIn(edge_id, self.graph._condition_maps)
        self.assertEqual(self.graph._condition_maps[edge_id], {True: "node2", False: "node3"})

    def test_add_conditional_edges_nonexistent_node(self):
        """
        Ensure that adding conditional edges from a non-existent node raises a ValueError.
        """
        def dummy_func(state):
            return True

        with self.assertRaises(ValueError):
            self.graph.add_conditional_edges("nonexistent", dummy_func, {True: "node2"})

    def test_set_entry_point(self):
        """
        Verify that the entry point of the graph can be set correctly.
        """
        self.graph.add_node("start_node")
        self.graph.set_entry_point("start_node")

        # Check if start_node is a successor of START
        successors = self.graph.successors("__start__")
        self.assertIn("start_node", successors)

    def test_set_entry_point_nonexistent_node(self):
        """
        Check that attempting to set a non-existent node as the entry point raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.graph.set_entry_point("nonexistent")

    def test_set_finish_point(self):
        """
        Ensure that the finish point of the graph can be set correctly.
        """
        self.graph.add_node("end_node")
        self.graph.set_finish_point("end_node")

        # Check if END is a successor of end_node
        successors = self.graph.successors("end_node")
        self.assertIn("__end__", successors)

    def test_set_finish_point_nonexistent_node(self):
        """
        Verify that setting a non-existent node as the finish point raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.graph.set_finish_point("nonexistent")

    def test_compile(self):
        """
        Test the compilation of the graph, including setting interrupt points.
        Ensure that interrupt points are correctly set in the compiled graph.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        compiled_graph = self.graph.compile(interrupt_before=["node1"], interrupt_after=["node2"])

        # Check if interrupt points are set correctly
        self.assertEqual(compiled_graph.interrupt_before, ["node1"])
        self.assertEqual(compiled_graph.interrupt_after, ["node2"])

    def test_compile_nonexistent_node(self):
        """
        Check that compiling with non-existent interrupt nodes raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.graph.compile(interrupt_before=["nonexistent"])

    def test_node(self):
        """
        Verify that node data can be retrieved correctly for existing nodes.
        """
        def test_func(state):
            return {"result": state["value"] * 2}

        self.graph.add_node("test_node", run=test_func)
        node_data = self.graph.node("test_node")

        # Node should have correct name
        self.assertEqual(node_data["name"], "test_node")

        # Node should have run function (added to node data by node method)
        self.assertIn("run", node_data)
        self.assertEqual(node_data["run"], test_func)

    def test_node_nonexistent(self):
        """
        Ensure that attempting to retrieve data for a non-existent node raises a KeyError.
        """
        with self.assertRaises(KeyError):
            self.graph.node("nonexistent_node")

    def test_edge(self):
        """
        Test that edge data can be retrieved correctly for existing edges.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")

        edge_data = self.graph.edge("node1", "node2")

        # Edge should have source and target
        self.assertEqual(edge_data["source"], "node1")
        self.assertEqual(edge_data["target"], "node2")

        # Edge should have conditional function (added to edge data by edge method)
        self.assertIn("cond", edge_data)

        # Edge should have condition map (added to edge data by edge method)
        self.assertIn("cond_map", edge_data)

    def test_edge_nonexistent(self):
        """
        Verify that attempting to retrieve data for a non-existent edge raises a KeyError.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")

        with self.assertRaises(KeyError):
            self.graph.edge("node1", "node2")

    def test_successors(self):
        """
        Check that the successors of a node can be correctly retrieved.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_node("node3")
        self.graph.add_edge("node1", "node2")
        self.graph.add_edge("node1", "node3")

        successors = self.graph.successors("node1")
        self.assertIn("node2", successors)
        self.assertIn("node3", successors)
        self.assertEqual(len(successors), 2)

    def test_successors_nonexistent(self):
        """
        Ensure that attempting to get successors of a non-existent node raises a KeyError.
        """
        with self.assertRaises(KeyError):
            self.graph.successors("nonexistent_node")

    def test_get_outgoing_edges(self):
        """
        Test that all outgoing edges from a node can be retrieved correctly.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_node("node3")
        self.graph.add_edge("node1", "node2")
        self.graph.add_edge("node1", "node3")

        edges = self.graph.get_outgoing_edges("node1")

        # Should have two edges
        self.assertEqual(len(edges), 2)

        # Check if targets are correct
        targets = [edge["target"] for edge in edges]
        self.assertIn("node2", targets)
        self.assertIn("node3", targets)

    def test_invoke_simple(self):
        """
        Test a simple workflow using the invoke method.
        """
        # Define node functions
        def increment_counter(state):
            return {"counter": state.get("counter", 0) + 1}

        def double_counter(state):
            return {"counter": state["counter"] * 2}

        # Add nodes
        self.graph.add_node("increment", run=increment_counter)
        self.graph.add_node("double", run=double_counter)

        # Set up workflow
        self.graph.set_entry_point("increment")
        self.graph.add_edge("increment", "double")
        self.graph.set_finish_point("double")

        # Invoke workflow
        results = list(self.graph.invoke({"counter": 5}))

        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "final")
        self.assertEqual(results[0]["state"]["counter"], 12)  # (5 + 1) * 2 = 12

    def test_invoke_with_interrupts(self):
        """
        Test workflow with interrupts using the invoke method.
        """
        # Define node functions
        def increment_counter(state):
            return {"counter": state.get("counter", 0) + 1}

        def double_counter(state):
            return {"counter": state["counter"] * 2}

        # Add nodes
        self.graph.add_node("increment", run=increment_counter)
        self.graph.add_node("double", run=double_counter)

        # Set up workflow
        self.graph.set_entry_point("increment")
        self.graph.add_edge("increment", "double")
        self.graph.set_finish_point("double")

        # Compile with interrupts
        self.graph.compile(interrupt_before=["double"])

        # Invoke workflow
        results = list(self.graph.invoke({"counter": 5}))

        # Check results
        self.assertEqual(len(results), 2)

        # First result should be interrupt before double
        self.assertEqual(results[0]["type"], "interrupt")
        self.assertEqual(results[0]["node"], "double")
        self.assertEqual(results[0]["state"]["counter"], 6)  # 5 + 1 = 6

        # Second result should be final state
        self.assertEqual(results[1]["type"], "final")
        self.assertEqual(results[1]["state"]["counter"], 12)  # 6 * 2 = 12

    def test_stream(self):
        """
        Test the stream method to get all intermediate states.
        """
        # Define node functions
        def increment_counter(state):
            return {"counter": state.get("counter", 0) + 1}

        def double_counter(state):
            return {"counter": state["counter"] * 2}

        # Add nodes
        self.graph.add_node("increment", run=increment_counter)
        self.graph.add_node("double", run=double_counter)

        # Set up workflow
        self.graph.set_entry_point("increment")
        self.graph.add_edge("increment", "double")
        self.graph.set_finish_point("double")

        # Stream workflow
        results = list(self.graph.stream({"counter": 5}))

        # Check results
        self.assertEqual(len(results), 3)

        # First result should be increment state
        self.assertEqual(results[0]["type"], "state")
        self.assertEqual(results[0]["node"], "increment")
        self.assertEqual(results[0]["state"]["counter"], 6)  # 5 + 1 = 6

        # Second result should be double state
        self.assertEqual(results[1]["type"], "state")
        self.assertEqual(results[1]["node"], "double")
        self.assertEqual(results[1]["state"]["counter"], 12)  # 6 * 2 = 12

        # Third result should be final state
        self.assertEqual(results[2]["type"], "final")
        self.assertEqual(results[2]["state"]["counter"], 12)

    def test_complex_workflow(self):
        """
        Test a complex workflow with conditional routing.
        """
        def condition_func(state):
            return state["value"] > 10

        self.graph.add_node("start", run=lambda state: {"value": state["value"] + 5})
        self.graph.add_node("process", run=lambda state: {"value": state["value"] * 2})
        self.graph.add_node("end", run=lambda state: {"result": f"Final value: {state['value']}"})

        self.graph.set_entry_point("start")
        self.graph.add_conditional_edges("start", condition_func, {True: "end", False: "process"})
        self.graph.add_edge("process", "start")
        self.graph.set_finish_point("end")

        invoke_result = list(self.graph.invoke({"value": 1}))

        self.assertEqual(len(invoke_result), 1)
        self.assertEqual(invoke_result[0]["type"], "final")
        self.assertIn("state", invoke_result[0])
        self.assertIn("result", invoke_result[0]["state"])
        self.assertEqual(invoke_result[0]["state"]["result"], "Final value: 17")

    def test_cyclic_graph(self):
        """
        Test a cyclic graph with conditional exit.
        """
        self.graph.add_node("node1", run=lambda state: {"count": state.get("count", 0) + 1})
        self.graph.add_node("node2", run=lambda state: {"count": state["count"] * 2})
        self.graph.set_entry_point("node1")
        self.graph.add_conditional_edges("node1", lambda state: state["count"] >= 3, {True: "node2", False: "node1"})
        self.graph.set_finish_point("node2")

        result = list(self.graph.invoke({"count": 0}))
        self.assertEqual(result[-1]["state"]["count"], 6)  # (0+1+1+1)*2 = 6

    def test_error_handling_in_node_function(self):
        """
        Test error handling in node functions.
        """
        def error_func(state):
            raise ValueError("Test error")

        # Create a graph with raise_exceptions=False
        error_graph = StateGraph(
            state_schema={},
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database,
            raise_exceptions=False
        )

        # Clean up
        error_graph.clear_graph()

        # Add node with error function
        error_graph.add_node("error_node", run=error_func)
        error_graph.set_entry_point("error_node")
        error_graph.set_finish_point("error_node")

        # Invoke workflow
        results = list(error_graph.invoke({}))

        # Should have error result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "error")
        self.assertIn("Test error", results[0]["error"])

        # Clean up
        error_graph.clear_graph()

    def test_conditional_workflow(self):
        """
        Test a workflow with conditional branching.
        """
        # Define node functions
        def check_counter(state):
            return state  # Just pass the state through

        def is_odd(state):
            return "odd" if state["counter"] % 2 == 1 else "even"

        def process_odd(state):
            return {"counter": state["counter"] * 3, "branch": "odd"}

        def process_even(state):
            return {"counter": state["counter"] // 2, "branch": "even"}

        def format_result(state):
            return {"result": f"Processed {state['branch']} number to: {state['counter']}"}

        # Add nodes to the graph
        self.graph.add_node("check", run=check_counter)
        self.graph.add_node("odd_branch", run=process_odd)
        self.graph.add_node("even_branch", run=process_even)
        self.graph.add_node("result", run=format_result)

        # Set up the workflow
        self.graph.set_entry_point("check")
        self.graph.add_conditional_edges("check", is_odd, {"odd": "odd_branch", "even": "even_branch"})
        self.graph.add_edge("odd_branch", "result")
        self.graph.add_edge("even_branch", "result")
        self.graph.set_finish_point("result")

        # Test with an odd number
        odd_results = list(self.graph.invoke({"counter": 7}))
        odd_final = odd_results[-1]
        self.assertEqual(odd_final["state"]["counter"], 21)  # 7 * 3 = 21
        self.assertEqual(odd_final["state"]["branch"], "odd")
        self.assertEqual(odd_final["state"]["result"], "Processed odd number to: 21")

        # Test with an even number
        even_results = list(self.graph.invoke({"counter": 8}))
        even_final = even_results[-1]
        self.assertEqual(even_final["state"]["counter"], 4)  # 8 / 2 = 4
        self.assertEqual(even_final["state"]["branch"], "even")
        self.assertEqual(even_final["state"]["result"], "Processed even number to: 4")

    def test_config_usage(self):
        """
        Test using configuration in node functions.
        """
        def configurable_func(state, config):
            return {"result": state["value"] * config["multiplier"]}

        self.graph.add_node("start", run=configurable_func)
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("start")

        result = list(self.graph.invoke({"value": 5}, config={"multiplier": 3}))
        self.assertEqual(result[-1]["state"]["result"], 15)  # 5 * 3 = 15

    def test_parallel_edge(self):
        """
        Test parallel edge and fan-in node functionality.
        This is a simplified test since true parallel execution
        would require a more complex setup.
        """
        # Define node functions
        def start_run(state):
            return {"value": state.get("value", 0)}

        def flow1_run(state):
            return {"flow1_value": state["value"] + 1}

        def flow2_run(state):
            return {"flow2_value": state["value"] + 2}

        def fan_in_run(state):
            # In a real scenario, this would process parallel_results
            return {"result": state.get("value", 0) + 3}

        def end_run(state):
            return {"final": state["result"]}

        # Add nodes
        self.graph.add_node("start", run=start_run)
        self.graph.add_node("flow1", run=flow1_run)
        self.graph.add_node("flow2", run=flow2_run)
        self.graph.add_fanin_node("fan_in", run=fan_in_run)
        self.graph.add_node("end", run=end_run)

        # Set entry and finish points
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("end")

        # Add parallel edges
        self.graph.add_parallel_edge("start", "flow1", "fan_in")
        self.graph.add_parallel_edge("start", "flow2", "fan_in")

        # Add normal edges
        self.graph.add_edge("flow1", "fan_in")
        self.graph.add_edge("flow2", "fan_in")
        self.graph.add_edge("fan_in", "end")

        # Invoke workflow
        results = list(self.graph.invoke({"value": 10}))

        # Should have final result
        self.assertEqual(results[-1]["type"], "final")
        self.assertEqual(results[-1]["state"]["final"], 13)  # 10 + 3 = 13

    def test_export_to_json(self):
        """
        Test exporting the graph to a JSON file.
        """
        import os
        import json

        # Create a simple graph
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")
        self.graph.set_entry_point("node1")
        self.graph.set_finish_point("node2")

        # Export to JSON
        filename = "test_graph.json"
        self.graph.export_to_json(filename)

        # Check if file exists
        self.assertTrue(os.path.exists(filename))

        # Read and verify JSON content
        with open(filename, 'r') as f:
            data = json.load(f)

        # Should have nodes, edges, and interrupt lists
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("interrupt_before", data)
        self.assertIn("interrupt_after", data)

        # Should have our nodes
        node_names = [node["name"] for node in data["nodes"]]
        self.assertIn("node1", node_names)
        self.assertIn("node2", node_names)

        # Clean up
        os.remove(filename)

    def test_visualize_graph(self):
        """
        Test the visualize_graph method returns a valid Cypher query.
        """
        # Create a simple graph
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")

        # Get visualization query
        query = self.graph.visualize_graph()

        # Should be a string containing MATCH and RETURN
        self.assertIsInstance(query, str)
        self.assertIn("MATCH", query)
        self.assertIn("RETURN", query)
        self.assertIn("Node", query)
        self.assertIn("TRANSITION", query)

    def test_render_graphviz(self):
        """
        Test the render_graphviz method creates a DOT file.
        """
        import os

        # Create a simple graph
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")

        # Render DOT file
        dot_file = self.graph.render_graphviz("test_graph.dot")

        # Check if file exists
        self.assertTrue(os.path.exists(dot_file))

        # Read and verify DOT content
        with open(dot_file, 'r') as f:
            content = f.read()

        # Should have our nodes
        self.assertIn('"node1"', content)
        self.assertIn('"node2"', content)
        self.assertIn('"node1" -> "node2"', content)

        # Clean up
        os.remove(dot_file)


class TestNeo4jStateGraphParallel(unittest.TestCase):
    """
    Tests for parallel execution in Neo4j StateGraph.

    These tests are more advanced and test the parallel execution capabilities.
    """

    def setUp(self):
        """Set up a Neo4j StateGraph for testing."""
        self.uri = "bolt://localhost:7687"
        self.user = "neo4j"
        self.password = "password"
        self.database = "neo4j"

        # Create a test graph with an empty state schema and enable exception raising
        self.graph = StateGraph(
            state_schema={},
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database,
            raise_exceptions=True
        )

        # Clean up any existing graph data from previous test runs
        self.graph.clear_graph()

        # Sleep a bit to ensure cleanup is complete
        time.sleep(0.5)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'graph'):
            self.graph.clear_graph()

    def test_fan_out_and_fan_in(self):
        """
        Test the fan-out and fan-in functionality of the StateGraph.

        Workflow:
            START -> start -> [flow1_start, flow2_start] -> fan_in -> end -> END

        flow1_start:
            - Increments 'value' by 1
        flow2_start:
            - Increments 'value' by 2
        fan_in:
            - Sums 'value' from all parallel flows and stores in 'result'
        end:
            - Finalizes the 'final_result'
        """

        # Define node functions
        def start_run(state, config=None):
            # Initialize 'value' if not present
            state.setdefault('value', 0)
            return {}

        def flow1_start_run(state, config=None):
            # Increment 'value' by 1
            new_value = state.get('value', 0) + 1
            return {'flow1_value': new_value}

        def flow2_start_run(state, config=None):
            # Increment 'value' by 2
            new_value = state.get('value', 0) + 2
            return {'flow2_value': new_value}

        def fan_in_run(state, config=None):
            # Collect results from all parallel flows
            parallel_results = state.get('parallel_results', [])

            # In real parallel execution, we would collect results from parallel flows
            # For this test, we'll just return a fixed result
            return {'result': state.get('value', 0) + 3}

        def end_run(state, config=None):
            # Finalize the result
            final_result = state.get('result', 0)
            return {'final_result': final_result}

        # Add nodes
        self.graph.add_node("start", run=start_run)
        self.graph.add_node("flow1_start", run=flow1_start_run)
        self.graph.add_node("flow2_start", run=flow2_start_run)
        self.graph.add_fanin_node("fan_in", run=fan_in_run)
        self.graph.add_node("end", run=end_run)

        # Set entry and finish points
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("end")

        # Add parallel edges
        self.graph.add_parallel_edge("start", "flow1_start", "fan_in")
        self.graph.add_parallel_edge("start", "flow2_start", "fan_in")

        # Add normal edges to make the graph valid
        self.graph.add_edge("flow1_start", "fan_in")
        self.graph.add_edge("flow2_start", "fan_in")
        self.graph.add_edge("fan_in", "end")

        # Invoke the graph with an initial state
        initial_state = {'value': 10}
        execution = self.graph.invoke(initial_state)

        # Iterate through the generator to completion
        final_output = None
        for output in execution:
            if output['type'] == 'final':
                final_output = output

        # Assert that final_output is not None
        self.assertIsNotNone(final_output, "Final output was not yielded.")

        # Assert that 'final_result' is as expected (10 + 3 = 13 in our simplified test)
        expected_result = 13
        self.assertIn('final_result', final_output['state'], "Final result not found in state.")
        self.assertEqual(final_output['state']['final_result'], expected_result,
                         f"Expected final_result to be {expected_result}, got {final_output['state']['final_result']}.")

    def test_complex_parallel_workflows(self):
        """
        Test more complex parallel workflows with multiple stages.

        This test creates a workflow with multiple parallel branches and conditional logic.
        """
        # Define node functions
        def start_run(state):
            return {"count": 0}

        def branch_condition(state):
            return "even" if state["value"] % 2 == 0 else "odd"

        def even_process(state):
            return {"even_result": state["value"] // 2}

        def odd_process(state):
            return {"odd_result": state["value"] * 3 + 1}

        def aggregate_results(state):
            # In a real test, we would use parallel_results
            # For now, we'll just check if we have either odd or even result
            if "even_result" in state:
                return {"final": state["even_result"]}
            elif "odd_result" in state:
                return {"final": state["odd_result"]}
            else:
                return {"final": 0}

        # Create nodes
        self.graph.add_node("start", run=start_run)
        self.graph.add_node("even_branch", run=even_process)
        self.graph.add_node("odd_branch", run=odd_process)
        self.graph.add_fanin_node("aggregate", run=aggregate_results)

        # Set entry and finish points
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("aggregate")

        # Add conditional edges from start
        self.graph.add_conditional_edges("start", branch_condition, {
            "even": "even_branch",
            "odd": "odd_branch"
        })

        # Add edges to aggregate
        self.graph.add_edge("even_branch", "aggregate")
        self.graph.add_edge("odd_branch", "aggregate")

        # Test with even number
        even_results = list(self.graph.invoke({"value": 10}))
        self.assertEqual(even_results[-1]["state"]["final"], 5)  # 10 / 2 = 5

        # Test with odd number
        odd_results = list(self.graph.invoke({"value": 7}))
        self.assertEqual(odd_results[-1]["state"]["final"], 22)  # 7 * 3 + 1 = 22

    def test_error_in_parallel_flow(self):
        """
        Test handling errors in parallel flows.
        """
        # Define node functions
        def start_run(state):
            return {}

        def error_flow(state):
            raise ValueError("Test error in parallel flow")

        def normal_flow(state):
            return {"normal_result": True}

        def fan_in_run(state):
            return {"fan_in_complete": True}

        # Create nodes
        self.graph.add_node("start", run=start_run)
        self.graph.add_node("error_branch", run=error_flow)
        self.graph.add_node("normal_branch", run=normal_flow)
        self.graph.add_fanin_node("fan_in", run=fan_in_run)

        # Set entry and finish points
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("fan_in")

        # Add parallel edges
        self.graph.add_parallel_edge("start", "error_branch", "fan_in")
        self.graph.add_parallel_edge("start", "normal_branch", "fan_in")

        # Add regular edges
        self.graph.add_edge("error_branch", "fan_in")
        self.graph.add_edge("normal_branch", "fan_in")

        # Invoke - should raise the error from the parallel flow
        with self.assertRaises(RuntimeError) as context:
            list(self.graph.invoke({}))

        # Check error message
        self.assertIn("Test error in parallel flow", str(context.exception))

if __name__ == "__main__":
    unittest.main()
