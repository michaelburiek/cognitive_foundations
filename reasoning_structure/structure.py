import os
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import matplotlib.pyplot as plt
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import json_repair
from tqdm import tqdm
from scipy import stats
import numpy as np
import math
import colorsys
import pygraphviz as pgv


@dataclass
class Model:
    """Represents a model and tracks problems it has answered."""
    name: str
    problems: Set[str] = field(default_factory=set)
    correct_problems: Set[str] = field(default_factory=set)
    incorrect_problems: Set[str] = field(default_factory=set)
    
    def add_problem(self, problem_id: str, is_correct: bool):
        """Add a problem that this model has answered."""
        self.problems.add(problem_id)
        if is_correct:
            self.correct_problems.add(problem_id)
        else:
            self.incorrect_problems.add(problem_id)
    
    def get_accuracy(self) -> float:
        """Calculate overall accuracy."""
        if not self.problems:
            return 0.0
        return len(self.correct_problems) / len(self.problems)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Model):
            return self.name == other.name
        return False


@dataclass
class Problem:
    """Represents a problem/question with its metadata."""
    problem_id: str
    task_category: str = ""
    problem_type: str = ""
    modality: str = "text"
    correctness: Dict[str, bool] = field(default_factory=dict)  # model_name -> is_correct
    expected_elements: Dict[str, int] = field(default_factory=dict)
    
    def add_model_result(self, model_name: str, is_correct: bool):
        """Record a model's correctness on this problem."""
        self.correctness[model_name] = is_correct
    
    def get_correct_models(self) -> List[str]:
        """Get list of models that answered correctly."""
        return [m for m, correct in self.correctness.items() if correct]
    
    def get_incorrect_models(self) -> List[str]:
        """Get list of models that answered incorrectly."""
        return [m for m, correct in self.correctness.items() if not correct]
    
    def __hash__(self):
        return hash(self.problem_id)
    
    def __eq__(self, other):
        if isinstance(other, Problem):
            return self.problem_id == other.problem_id
        return False


@dataclass
class Edge:
    """Represents a connection between two element nodes."""
    node_a: str
    node_b: str
    edge_type: str  # "next", "contains", or "parallel"
    weight: int = 0
    occurrences: List[Tuple[str, str]] = field(default_factory=list)  # (model_name, problem_id)
    
    def __hash__(self):
        return hash((self.node_a, self.node_b, self.edge_type))
    
    def add_occurrence(self, model_name: str, problem_id: str):
        """Record an occurrence of this edge in a specific model's trace for a problem."""
        self.occurrences.append((model_name, problem_id))
        self.weight += 1
    
    def get_occurrences_by_model(self, model_name: str) -> int:
        """Count occurrences for a specific model."""
        return sum(1 for m, _ in self.occurrences if m == model_name)
    
    def get_occurrences_by_problem(self, problem_id: str) -> int:
        """Count occurrences for a specific problem."""
        return sum(1 for _, p in self.occurrences if p == problem_id)


@dataclass
class ElementNode:
    """Represents a element node with its spans and connections."""
    element: str
    spans: List[Tuple[int, int, str, str]] = field(default_factory=list)  # (start, end, model_name, problem_id)
    total_span_length: int = 0
    frequency: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))  # (model, problem) -> count
    connections: Dict[str, Dict[str, List[Edge]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    
    def add_span(self, start: int, end: int, model_name: str, problem_id: str):
        """Add a span to this element node."""
        self.spans.append((start, end, model_name, problem_id))
        self.total_span_length += (end - start)
        self.frequency[(model_name, problem_id)] += 1
    
    def get_total_occurrences(self) -> int:
        """Get total number of occurrences across all model-problem pairs."""
        return sum(self.frequency.values())
    
    def get_avg_span_length(self) -> float:
        """Calculate average span length."""
        if not self.spans:
            return 0.0
        return self.total_span_length / len(self.spans)
    
    def add_connection(self, other_element: str, edge: Edge):
        """Add a connection to another element."""
        self.connections[other_element][edge.edge_type].append(edge)


class SpanTree:
    """Builds and manages a DAG of element relationships."""
    
    def __init__(self, overlap_threshold: float = 0.8, parallel_threshold: int = 20):
        """
        Initialize SpanTree.
        
        Args:
            overlap_threshold: Minimum overlap ratio for "parallel" relationship (0.0-1.0)
        """
        self.nodes: Dict[str, ElementNode] = {}
        # edges is a nested dict: (node_a, node_b) -> {edge_type: Edge}
        self.edges: Dict[Tuple[str, str], Dict[str, Edge]] = defaultdict(dict)
        self.overlap_threshold = overlap_threshold
        self.parallel_threshold = parallel_threshold
        self.trace_elements: Dict[Tuple[str, str], List[Tuple[str, int, int]]] = defaultdict(list)  # (model, problem) -> elements
        
        self.models: Dict[str, Model] = {}
        self.problems: Dict[str, Problem] = {}
    
    def get_all_edges(self) -> List[Edge]:
        """Get a flat list of all edges (for backward compatibility)."""
        edges = []
        for edge_types in self.edges.values():
            edges.extend(edge_types.values())
        return edges
    
    def get_edge(self, node_a: str, node_b: str, edge_type: str) -> Optional[Edge]:
        """Get a specific edge by its nodes and type."""
        edge_key = (node_a, node_b)
        if edge_key in self.edges:
            return self.edges[edge_key].get(edge_type)
        return None
    
    def load_element_files(self, directory: str, id2type: dict, target_type: str = None):
        """Load all element JSON files from a directory in parallel."""
        json_files = glob.glob(os.path.join(directory, '*.json'))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {directory}")
        
        print(f"Loading {len(json_files)} element files in parallel...")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()*1.5) as executor:
            future_to_file = {
                executor.submit(
                    self._process_single_file, 
                    json_file,
                    id2type, 
                    target_type
                ): json_file 
                for json_file in json_files
            }
            
            for future in as_completed(future_to_file):
                json_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        model_name, trace_elements, node_data, problems_data, models_data = result
                        
                        # Thread-safe: Consolidate in main thread
                        
                        # 1. Add/update problems
                        for problem_id, problem_info in problems_data.items():
                            if problem_id not in self.problems:
                                self.problems[problem_id] = Problem(
                                    problem_id=problem_id,
                                    task_category=problem_info['task_category'],
                                    problem_type=problem_info['problem_type'],
                                    modality=problem_info['modality'],
                                    expected_elements=problem_info['expected_elements']
                                )
                            # Add model result to problem
                            for model, is_correct in problem_info['model_results'].items():
                                self.problems[problem_id].add_model_result(model, is_correct)
                        
                        # 2. Add/update models
                        for model, model_info in models_data.items():
                            if model not in self.models:
                                self.models[model] = Model(name=model)
                            # Add problem results to model
                            for prob_id, is_correct in model_info['problem_results'].items():
                                self.models[model].add_problem(prob_id, is_correct)
                        
                        # 3. Add trace elements
                        for trace_key, elements in trace_elements.items():
                            self.trace_elements[trace_key].extend(elements)
                        
                        # 4. Add node data
                        for element_name, spans in node_data.items():
                            if element_name not in self.nodes:
                                self.nodes[element_name] = ElementNode(element=element_name)
                            node = self.nodes[element_name]
                            for start, end, model, problem in spans:
                                node.add_span(start, end, model, problem)
                        
                        print(f"  ✓ Loaded {model_name}")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {json_file}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Loaded {len(self.nodes)} elements")
        print(f"Loaded {len(self.models)} models")
        print(f"Loaded {len(self.problems)} problems")
        print(f"Loaded {len(self.trace_elements)} traces")
        self._build_relationships()


    def _process_single_file(self, json_file: str, id2type: dict, target_type: str = None) -> Optional[Tuple]:
        """
        Process a single model file.
        
        Returns all data WITHOUT modifying shared state (thread-safe).
        """
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json_repair.load(f)
        
        trace_elements = defaultdict(list)
        node_data = defaultdict(list)  # element_name -> [(start, end, model, problem), ...]
        problems_data = {}  # problem_id -> {task_category, problem_type, modality, model_results}
        models_data = defaultdict(lambda: {'problem_results': {}})  # model -> {problem_results: {prob_id: is_correct}}
        
        model_name = None  # Will be set from first question
        
        for question_key, question_data in data.items():
            #try:
            problem_id = str(question_data.get('question_id', question_key))
            if ('hier_' in problem_id) or ('graph_' in problem_id):
                continue
            current_model_name = question_data['model_name']
            
            # Set model_name on first iteration
            if model_name is None:
                model_name = current_model_name
            # Verify all questions in file are from same model
            elif model_name != current_model_name:
                print(f"  ⚠ Warning: Multiple models in {json_file}: {model_name} vs {current_model_name}")
                # Use the first model name seen
            
            task_category = question_data['task']

            problem_types = question_data['problem_type']
            if problem_types is None:
                problem_type = None
            elif type(problem_types) == str and not problem_types.isnumeric():
                problem_type = problem_types
            else:
                info = [int(i) if i.isnumeric() else -1 for i in problem_types]
                problem_type = id2type[stats.mode(np.array(info)).mode]

            # Filter by target type if specified
            if target_type and (problem_type != target_type):
                continue

            modality = question_data.get('modality', 'text')
            if 'image' in json_file.lower():
                correctness = True
                modality = question_data['correctness']
            elif 'audio' in json_file.lower():
                correctness = True
                modality = "audio"
            else:
                correctness = question_data['correctness']

            # Store problem data (will be created/updated in main thread)
            if problem_id not in problems_data:
                problems_data[problem_id] = {
                    'task_category': task_category,
                    'problem_type': problem_type,
                    'modality': modality,
                    'model_results': {},
                    'expected_elements': {}
                }
            problems_data[problem_id]['model_results'][model_name] = correctness
            
            # Store model data (will be created/updated in main thread)
            models_data[model_name]['problem_results'][problem_id] = correctness

            # Process element annotations
            for element_name, element_info in question_data['capability_annotation'].items():
                # Only consider if score >= 2
                if ('score' not in element_info) or (element_info['score'] < 2):
                    continue
                
                if ('element_expectation' not in element_info) or element_info['element_expectation'] is None:
                    element_expectation = None
                else:
                    element_expectation = int(element_info['element_expectation']) if (type(element_info['element_expectation']) == int) or element_info['element_expectation'].isnumeric() else 0
                    problems_data[problem_id]['expected_elements'][element_name] = element_expectation

                for span_group in element_info['spans']:
                    if not isinstance(span_group, list) or len(span_group) != 2:
                        continue

                    start, end = span_group

                    # Validate span values
                    if isinstance(start, str):
                        if not start.isnumeric():
                            continue
                        start = int(start)
                    if isinstance(end, str):
                        if not end.isnumeric():
                            continue
                        end = int(end)
                    
                    # Store node data for main thread to process
                    node_data[element_name].append((start, end, model_name, problem_id))
                    
                    # Store trace element
                    trace_key = (model_name, problem_id)
                    trace_elements[trace_key].append((element_name, start, end))
            
            # except Exception as e:
            #     print(f"  ✗ Error processing question {question_key} in {json_file}: {e}")
            #     continue
        
        if model_name is None:
            return None
        
        return (model_name, trace_elements, node_data, problems_data, models_data)


    def _build_relationships(self):
        """Build relationships between elements based on span positions."""
        print("\nBuilding relationships in parallel...")

        trace_keys = list(self.trace_elements.keys())

        all_edges = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()*1.5) as executor:
            futures = {
                executor.submit(self._process_trace_relationships, trace_key): trace_key 
                for trace_key in trace_keys
            }
            
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing traces")
                
            for future in iterator:
                try:
                    edges = future.result()
                    all_edges.extend(edges)
                except Exception as e:
                    trace_key = futures[future]
                    print(f"  ✗ Error processing trace {trace_key}: {e}")
                    import traceback
                    traceback.print_exc()

        # Consolidate edges using the new dictionary structure
        for edge_data in all_edges:
            element_a, element_b, edge_type, model_name, problem_id = edge_data
            edge_key = (element_a, element_b)

            # Check if this edge type already exists
            if edge_type not in self.edges[edge_key]:
                # Create new edge
                edge = Edge(node_a=element_a, node_b=element_b, edge_type=edge_type)
                self.edges[edge_key][edge_type] = edge
            else:
                # Use existing edge
                edge = self.edges[edge_key][edge_type]

            edge.add_occurrence(model_name, problem_id)

            # Add connections to nodes
            self.nodes[element_a].add_connection(element_b, edge)
            if edge_type != "next":
                self.nodes[element_b].add_connection(element_a, edge)

        total_edges = sum(len(edge_types) for edge_types in self.edges.values())
        total_occurrences = sum(edge.weight for edge_types in self.edges.values() for edge in edge_types.values())
        
        print(f"Created {len(self.edges)} unique directed edges")
        print(f"Created {total_edges} unique edge types")
        print(f"Total edge occurrences: {total_occurrences}")


    def _process_trace_relationships(self, trace_key: Tuple[str, str]) -> List[Tuple]:
        """Process relationships for a single trace."""
        model_name, problem_id = trace_key
        elements_spans = self.trace_elements[trace_key]
        
        # Sort by start position, then by span length (descending)
        sorted_spans = sorted(elements_spans, key=lambda x: (x[1], -(x[2] - x[1])))
        
        edges = []
        for i, (element_a, start_a, end_a) in enumerate(sorted_spans):
            has_next = False
            for j, (element_b, start_b, end_b) in enumerate(sorted_spans):
                if i >= j:
                    continue
                
                edge_type = self._determine_relationship(start_a, end_a, start_b, end_b)
                
                # Refine "next" edges to only edges between a and b which have no other intermediate 'next' edge that is separate

                if edge_type == "next":
                    if has_next and (start_b >= end_a):
                        between = True
                        break
                    if has_next:
                        between = True
                        edge_type = None
                        break
                    else:
                        has_next = True
                        between = False
                
                if edge_type:
                    edges.append((element_a, element_b, edge_type, model_name, problem_id))
        
        return edges


    def _determine_relationship(self, start_a: int, end_a: int, start_b: int, end_b: int) -> Optional[str]:
        """Determine the relationship type between two spans."""
        if ((end_b - start_b) == 0) or ((end_a - start_a) == 0):
            return None

        if (abs(start_b - start_a) + abs(end_b - end_a)) <= self.parallel_threshold:
            return "parallel"

        if start_b <= end_a:
            if end_a <= end_b:
                overlap = (end_a - start_b)/(end_b - start_b)
                if overlap >= self.overlap_threshold:
                    return "contains"
                else:
                    return "next"
            else: # end_a > end_b
                overlap = (end_b - start_b) / (end_a - start_a)
                if overlap >= self.overlap_threshold:
                    return "parallel"
                else:
                    return "next"
        else: # end_a < start_b
            return "next"
    

    # ===== ANALYTICAL FUNCTIONS =====

    def compute_npmi_node(self, problem_type: str = None, all_elements: list = None) -> Dict:
        """
        Compute normalized pointwise mutual information (NPMI) between each element 
        and successful reasoning traces.
        
        NPMI(element, success) = PMI(element, success) / -log(P(element, success))
        where PMI(element, success) = log(P(element, success) / (P(element) * P(success)))
        
        NPMI ranges from -1 to 1:
        - 1: perfect positive association (element always appears with success)
        - 0: independence (element and success are unrelated)
        - -1: perfect negative association (element never appears with success)
        
        Args:
            problem_type: Filter by specific problem type (None for all)
            
        Returns:
            Dictionary with NPMI scores and component probabilities for each element
        """
        element_and_success = Counter({b: 0 for b in all_elements})  # Count of traces with both element and success
        element_only = Counter({b: 0 for b in all_elements})  # Count of traces with element (success or not)
        success_count = 0
        total_traces = 0
        
        for (model_name, problem_id), elements in self.trace_elements.items():
            if problem_id not in self.problems:
                continue
            
            problem = self.problems[problem_id]
            
            # Filter by problem type if specified
            if problem_type and problem.problem_type != problem_type:
                continue
            
            total_traces += 1
            is_correct = problem.correctness.get(model_name, False)
            
            if is_correct:
                success_count += 1
            
            element_names = set([b[0] for b in elements])
            
            # Count element occurrences
            element_only.update(element_names)
            
            # Count joint occurrences (element AND success)
            if is_correct:
                element_and_success.update(element_names)
        
        if total_traces == 0:
            return {}
        
        # Compute probabilities and NPMI
        p_success = success_count / total_traces
        
        npmi_scores = {}
        
        for element in all_elements:
            p_element = element_only[element] / total_traces
            p_element_and_success = element_and_success.get(element, 0) / total_traces
            
            # Avoid log(0) errors
            if p_element_and_success == 0 or p_element == 0 or p_success == 0:
                npmi_scores[element] = {
                    'npmi': -1.0,  # Minimum value indicates no co-occurrence
                    'pmi': float('-inf'),
                    'p_element': p_element,
                    'p_success': p_success,
                    'p_element_and_success': p_element_and_success
                }
                continue
            
            # PMI = log2(P(element, success) / (P(element) * P(success)))
            pmi = math.log2(p_element_and_success / (p_element * p_success))
            
            # NPMI = PMI / -log2(P(element, success))
            # Normalization ensures NPMI is in [-1, 1]
            npmi = pmi / (-math.log2(p_element_and_success))
            
            npmi_scores[element] = {
                'npmi': npmi,
                'pmi': pmi,
                'p_element': p_element,
                'p_success': p_success,
                'p_element_and_success': p_element_and_success,
                'count_element': element_only[element],
                'count_success_with_element': element_and_success.get(element, 0)
            }
        
        return npmi_scores


    def compute_npmi_edge(self, problem_type: str = None) -> Dict:
        """
        Compute normalized pointwise mutual information (NPMI) between each edge 
        and successful reasoning traces.
        
        Uses pre-built edges from SpanTree.edges instead of rebuilding them.
        
        NPMI(edge, success) = PMI(edge, success) / -log(P(edge, success))
        where PMI(edge, success) = log(P(edge, success) / (P(edge) * P(success)))
        
        NPMI ranges from -1 to 1:
        - 1: perfect positive association (edge always appears with success)
        - 0: independence (edge and success are unrelated)
        - -1: perfect negative association (edge never appears with success)
        
        Args:
            problem_type: Filter by specific problem type (None for all)
            
        Returns:
            Dictionary mapping (node_a, node_b, edge_type) -> NPMI statistics
        """
        edge_and_success = Counter()  # Count of traces with both edge and success
        edge_only = Counter()  # Count of traces with edge (success or not)
        success_count = 0
        total_traces = 0
        
        # Build a set of valid traces for this problem type
        valid_traces = set()
        
        for (model_name, problem_id) in self.trace_elements.keys():
            if problem_id not in self.problems:
                continue
            
            problem = self.problems[problem_id]
            
            # Filter by problem type if specified
            if problem_type and problem.problem_type != problem_type:
                continue
            
            valid_traces.add((model_name, problem_id))
            total_traces += 1
            
            is_correct = problem.correctness.get(model_name, False)
            if is_correct:
                success_count += 1
        
        if total_traces == 0:
            return {}
        
        # Use pre-built edges from SpanTree
        for edge_key, edge_types in self.edges.items():
            node_a, node_b = edge_key
            
            for edge_type, edge in edge_types.items():
                # Count occurrences in valid traces
                trace_key = (node_a, node_b, edge_type)
                
                for model_name, problem_id in edge.occurrences:
                    # Skip if not in our filtered set
                    if (model_name, problem_id) not in valid_traces:
                        continue
                    
                    edge_only[trace_key] += 1
                    
                    # Check if this trace was successful
                    if self.problems[problem_id].correctness.get(model_name, False):
                        edge_and_success[trace_key] += 1
        
        # Compute probabilities and NPMI
        p_success = success_count / total_traces
        
        npmi_scores = {}
        
        for edge_key in edge_only.keys():
            p_edge = edge_only[edge_key] / total_traces
            p_edge_and_success = edge_and_success.get(edge_key, 0) / total_traces
            
            # Avoid log(0) errors
            if p_edge_and_success == 0 or p_edge == 0 or p_success == 0:
                npmi_scores[edge_key] = {
                    'npmi': -1.0,  # Minimum value indicates no co-occurrence
                    'pmi': float('-inf'),
                    'p_edge': p_edge,
                    'p_success': p_success,
                    'p_edge_and_success': p_edge_and_success
                }
                continue
            
            # PMI = log2(P(edge, success) / (P(edge) * P(success)))
            pmi = math.log2(p_edge_and_success / (p_edge * p_success))
            
            # NPMI = PMI / -log2(P(edge, success))
            npmi = pmi / (-math.log2(p_edge_and_success))
            
            npmi_scores[edge_key] = {
                'npmi': npmi,
                'pmi': pmi,
                'p_edge': p_edge,
                'p_success': p_success,
                'p_edge_and_success': p_edge_and_success,
                'count_edge': edge_only[edge_key],
                'count_success_with_edge': edge_and_success.get(edge_key, 0)
            }
        
        return npmi_scores

# ----------------------------------------------------------------------------------------------------------------------
# Methods for identifying the most common/consensus tree structure
# from multiple successful reasoning traces.

class ConsensusTreeFinder:
    """Find consensus element tree structures from multiple successful traces."""
    
    def __init__(self, span_tree):
        """
        Initialize with a SpanTree instance.
        
        Args:
            span_tree: SpanTree instance with loaded data
        """
        self.span_tree = span_tree

    def construct_consensus_tree(self, problem_type: str, max_nodes: int = 10, success_only: bool = True, dynamic: bool = False) -> nx.DiGraph:
        """
        Construct a consensus tree of elements for a given problem type based on NPMI.
        
        The algorithm:
        1. Find the element with highest NPMI that appears first (earliest start) in successful traces
        2. From current element A, find the edge (A->B) with highest NPMI (any type: contains/next/parallel)
        3. Add edge to graph and move to element B
        4. Repeat until max_nodes reached or no more valid edges
        
        Args:
            problem_type: The problem type to analyze
            max_nodes: Maximum number of nodes to include in the consensus tree
            
        Returns:
            NetworkX DiGraph representing the consensus reasoning path
        """
        # Compute NPMI for elements and edges
        element_npmi = self.span_tree.compute_npmi_node(problem_type=problem_type, 
                                        all_elements=list(self.span_tree.nodes.keys()))
        edge_npmi = self.span_tree.compute_npmi_edge(problem_type=problem_type)
        
        if not element_npmi or not edge_npmi:
            print("No NPMI data available")
            return nx.DiGraph()
        
        # Get successful traces for this problem type
        traces = self._get_all_traces_type(problem_type=problem_type, success_only=success_only)
        
        if not traces:
            print("No traces found")
            return nx.DiGraph()
        
        # Find elements that appear first (earliest start) in successful traces
        first_elements = Counter()
        
        for trace_key in traces:
            if trace_key not in self.span_tree.trace_elements:
                continue
            
            elements = self.span_tree.trace_elements[trace_key]
            if not elements:
                continue
            
            # Sort by start position and get the first element
            sorted_elements = sorted(elements, key=lambda x: (x[1], -(x[2]-x[1])))
            first_element = sorted_elements[0][0]
            first_elements[first_element] += 1
        
        if not first_elements:
            print("No first elements found")
            return nx.DiGraph()
        
        if success_only:
            # Pick the first element with highest NPMI
            candidate_first = [(b, element_npmi[b]['npmi']) 
                            for b in first_elements.keys() 
                            if b in element_npmi]
        else:
            # Pick the first element with highest count
            candidate_first = [(b, first_elements[b]) 
                            for b in first_elements.keys() 
                            if b in element_npmi]
        
        if not candidate_first:
            print("No valid first elements with NPMI scores")
            return nx.DiGraph()
        
        # Start with the element that has highest NPMI (or overall frequency if all traces) among first elements
        current_element = max(candidate_first, key=lambda x: x[1])[0]
        
        # Build consensus graph
        consensus_graph = nx.DiGraph()
        consensus_graph.add_node(current_element, 
                                npmi=element_npmi[current_element]['npmi'],
                                is_start=True)
        
        visited = {current_element}
        
        print(f"Starting consensus tree with: {current_element} (NPMI: {element_npmi[current_element]['npmi']:.3f}, Node Prob: {element_npmi[current_element]['p_element']:.3f})")
        
        # Iteratively add nodes based on highest NPMI edges
        while (len(consensus_graph.nodes()) < max_nodes):
            # Find all outgoing edges from current element
            candidate_edges = []
            
            for (node_a, node_b, edge_type), npmi_data in edge_npmi.items():
                # Must originate from current element
                if node_a != current_element:
                    continue
                
                # Skip self-loops
                if node_a == node_b:
                    continue
                
                # Skip if target already visited (avoid cycles)
                if node_b in visited:
                    continue
                
                candidate_edges.append((node_b, edge_type, npmi_data['npmi'], npmi_data['p_edge']))
            
            if not candidate_edges:
                print(f"No more valid edges from {current_element}")
                break
            
            # Pick edge with highest NPMI
            if success_only:
                next_element, edge_type, edge_npmi_score, edge_prob_score = max(candidate_edges, key=lambda x: x[2])
            else:
                next_element, edge_type, edge_npmi_score, edge_prob_score = max(candidate_edges, key=lambda x: x[3])

            if success_only and dynamic and (edge_npmi_score <= 0):
                break
            
            # Add to graph
            consensus_graph.add_node(next_element, 
                                    npmi=element_npmi[next_element]['npmi'],
                                    p_element=element_npmi[next_element]['p_element'])
            consensus_graph.add_edge(current_element, next_element, 
                                    edge_type=edge_type,
                                    npmi=edge_npmi_score,
                                    p_edge=edge_prob_score)
            
            print(f"  Added edge: {current_element} --[{edge_type}]--> {next_element} "
                  f"(Edge NPMI: {edge_npmi_score:.3f}, Edge Prob: {edge_prob_score:.3f}, Node NPMI: {element_npmi[next_element]['npmi']:.3f}, Node Prob: {element_npmi[next_element]['p_element']:.3f}")
            
            visited.add(next_element)
            current_element = next_element
        
        print(f"\nConsensus tree complete: {len(consensus_graph.nodes())} nodes, {len(consensus_graph.edges())} edges")
        
        return consensus_graph

    def _get_all_traces_type(self, problem_type: str, success_only: bool = True) -> List[Tuple[str, str]]:
        """Get all successful (model, problem) traces for a problem type."""

        all_traces = []

        for problem_id, problem_info in self.span_tree.problems.items():
            if problem_info.problem_type != problem_type:
                continue
            
            if success_only:
                models = problem_info.get_correct_models()
            else:
                models = list(problem_info.correctness.keys())
            
            for model_name in models:
                if (model_name, problem_id) in self.span_tree.trace_elements:
                    all_traces.append((model_name, problem_id))
        
        return all_traces
    
    def visualize_semantic_consensus_graph(self,
                                           graph: nx.DiGraph,
                                           problem_type: str,
                                           max_nodes: int,
                                           output_file: str = "consensus_tree.png",
                                           font_name: str = "Nimbus Sans") -> None:
        """
        Improved visualization for reasoning-element graphs.
        - Adds a title and legend.
        - Uses consistent, categorical colors for element nodes.
        - Sequential edges define vertical flow (top → bottom).
        - Modern aesthetic with light fill and dark borders/fonts.
        """

        # --- 1. Setup & Directory ---
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if graph.number_of_nodes() == 0:
            print("Empty graph.")
            return

        # --- 2. Define Element Color Map (Consistent Coloring) ---
        all_elements = [
            'self-awareness', 'self-evaluation', 'logical-coherence', 
            'compositionality', 'sequential-organization', 'selective-attention', 
            'forward-chaining', 'causal-organization', 'knowledge-structure-alignment', 
            'strategy-selection', 'goal-management', 'ordinal-organization', 
            'temporal-organization', 'context-alignment', 'verification', 
            'backtracking', 'conceptual-level-processing', 
            'decomposition-and-integration', 'representational-restructuring', 
            'abstraction', 'backward-chaining', 'productivity', 
            'hierarchical-organization', 'adaptive-detail-management', 
            'pattern-recognition', 'spatial-organization', 'network-organization', 
            'context-awareness'
        ]
        
        cmap_tab20 = plt.cm.get_cmap("Set2")
        cmap_tab20b = plt.cm.get_cmap("Set3")
        colors_list = [cmap_tab20(i) for i in range(cmap_tab20.N)] + \
                    [cmap_tab20b(i) for i in range(cmap_tab20b.N)]
        
        element_color_map = {}
        for i, element in enumerate(all_elements):
            r, g, b, _ = colors_list[i % len(colors_list)]
            element_color_map[element] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        
        default_color = "#AAAAAA"

        # --- 3. Compute Node "Levels" (same as before) ---
        seq_edges = [(u, v) for u, v, d in graph.edges(data=True)
                    if d.get("edge_type") == "next"]

        seq_graph = nx.DiGraph()
        seq_graph.add_nodes_from(graph.nodes())
        seq_graph.add_edges_from(seq_edges)

        if seq_graph.number_of_edges() > 0:
            try:
                levels = nx.topological_generations(seq_graph)
                level_map = {}
                for depth, layer in enumerate(levels):
                    for node in layer:
                        level_map[node] = depth
            except nx.NetworkXUnfeasible:
                level_map = {n: 0 for n in graph.nodes()}
        else:
            level_map = {n: 0 for n in graph.nodes()}

        # --- 4. Use Graphviz DOT Layout ---

        current_nodes = graph.number_of_nodes()
        title_label = f"""<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4">
        <TR><TD COLSPAN="2"><B>{problem_type} (Nodes: {current_nodes} / {max_nodes})</B></TD></TR>
        <TR><TD COLSPAN="2"><HR></HR></TD></TR>
        <TR><TD ALIGN="RIGHT">Next:</TD><TD ALIGN="LEFT"><FONT COLOR="#3498db">━━━</FONT> (solid)</TD></TR>
        <TR><TD ALIGN="RIGHT">Contains:</TD><TD ALIGN="LEFT"><FONT COLOR="#e74c3c">‑ ‑ ‑</FONT> (dashed)</TD></TR>
        <TR><TD ALIGN="RIGHT">Parallel:</TD><TD ALIGN="LEFT"><FONT COLOR="#2ecc71">. . .</FONT> (dotted)</TD></TR>
        </TABLE>
        >"""

        A = pgv.AGraph(directed=True, 
                        rankdir="TB",
                        labelloc="t",
                        label=title_label,
                        fontsize="14",
                        pad="0.5",
                        splines="spline") # Use curved lines

        if font_name in A.graph_attr['fontname']:
            A.graph_attr['fontname'] = font_name
            A.node_attr['fontname'] = font_name
            A.edge_attr['fontname'] = font_name
        else:
            pass

        
        for node in graph.nodes():
            # Get the base color
            base_color = element_color_map.get(node, default_color)
            # Generate light fill and dark border/font
            fill_color, font_and_border_color = self._get_modern_colors(base_color)

            A.add_node(
                node,
                label=node,
                shape="circle",
                style="filled",
                fillcolor=fill_color,
                color=font_and_border_color,
                fontcolor=font_and_border_color,
                penwidth="2.5",
                fontsize="12"
            )

        # Add edges
        for u, v, d in graph.edges(data=True):
            e_type = d.get("edge_type", "next")
            
            color = {"next": "#3498db",
                    "contains": "#e74c3c",
                    "parallel": "#2ecc71"}[e_type]

            style = {"next": "solid",
                    "contains": "dashed",
                    "parallel": "dotted"}[e_type]

            A.add_edge(u, v,
                    color=color,
                    penwidth="2.5",
                    style=style,
                    arrowsize="1.0")

        # Enforce ranks so sequential layers align
        layers = {}
        for n, lvl in level_map.items():
            layers.setdefault(lvl, []).append(n)

        for lvl_nodes in layers.values():
            A.add_subgraph(lvl_nodes, rank="same")

        # Save layout
        A.layout(prog="dot")
        A.draw(output_file)
        print(f"Saved graph to: {output_file}")
    
    def _get_modern_colors(self, base_hex: str):
        """
        Generates a light fill color and a dark border/font color from a 
        single base hex color.
        """
        # 1. Convert hex to RGB normalized to [0, 1]
        base_hex = base_hex.lstrip('#')
        r, g, b = [int(base_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        
        # 2. Convert RGB to HLS (Hue, Lightness, Saturation)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        
        # 3. Create light (fill) and dark (border/font) versions
        # Light version: increase lightness (e.g., 80% towards white)
        light_l = l + (1.0 - l) * 0.8
        # Dark version: decrease lightness (e.g., 60% towards black)
        dark_l = l * 0.6
        
        # 4. Convert back to RGB
        light_r, light_g, light_b = colorsys.hls_to_rgb(h, light_l, s)
        dark_r, dark_g, dark_b = colorsys.hls_to_rgb(h, dark_l, s)
        
        # 5. Helper to convert normalized RGB back to hex
        def to_hex(nr, ng, nb):
            return f"#{int(nr*255):02x}{int(ng*255):02x}{int(nb*255):02x}"
            
        fill_color = to_hex(light_r, light_g, light_b)
        border_color = to_hex(dark_r, dark_g, dark_b)

        if dark_r > 0.95:
            border_color = "#172849" # Dark blue
        
        return fill_color, border_color