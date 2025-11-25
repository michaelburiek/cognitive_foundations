import os
import json
import asyncio
import argparse
from tqdm.asyncio import tqdm as async_tqdm
from pydantic import BaseModel
import json_repair
from openai import AsyncOpenAI
import networkx as nx
import numpy as np
import random


class ReasoningTrace(BaseModel):
    final_answer: str


class ElementGuidedReasoning:
    """Generate reasoning traces guided by problem-type-specific element graphs."""
    
    def __init__(self, args, span_tree, consensus_finder, problem_type_graphs):
        self.args = args
        self.span_tree = span_tree
        self.consensus_finder = consensus_finder
        self.problem_type_graphs = problem_type_graphs
        self.path_to_question_info = args.path_to_question_info
        self.sampled_questions = {}
    
    def sample_questions_per_type(self):
        """Sample questions for each problem type, balancing successful/unsuccessful traces."""
        print("\nSampling questions per problem type...")

        with open(self.args.path_to_question_info, 'r') as f:
            eval_json = json.load(f)
        all_question_info = {str(question_info['question_id']): question_info['question'] for question_info in eval_json if question_info['model'] == 'Qwen3-8B'}
        
        for problem_type in self.problem_type_graphs.keys():
            # Get all problems of this type
            problems_of_type = [
                (pid, problem) for pid, problem in self.span_tree.problems.items()
                if (problem.problem_type == problem_type) and (problem.problem_id in all_question_info)
            ]
            
            # Separate by success/failure for the target model
            successful = []
            unsuccessful = []
            
            for pid, problem in problems_of_type:
                if self.args.model_name in problem.correctness:
                    if problem.correctness[self.args.model_name]:
                        successful.append(pid)
                    else:
                        unsuccessful.append(pid)
            
            print(f"\nProblem type: {problem_type}")
            print(f"  Total problems: {len(problems_of_type)}")
            print(f"  Successful: {len(successful)}")
            print(f"  Unsuccessful: {len(unsuccessful)}")
            
            # Sample with desired distribution
            n_samples = min(self.args.samples_per_type, len(problems_of_type))
            target_successful = n_samples // 2
            target_unsuccessful = n_samples - target_successful
            
            # Adjust if we don't have enough of one type
            actual_successful = min(target_successful, len(successful))
            actual_unsuccessful = min(target_unsuccessful, len(unsuccessful))
            
            # If one category is short, try to take more from the other
            if actual_successful < target_successful:
                actual_unsuccessful = min(n_samples - actual_successful, len(unsuccessful))
            if actual_unsuccessful < target_unsuccessful:
                actual_successful = min(n_samples - actual_unsuccessful, len(successful))
            
            # Sample
            sampled_successful = random.sample(successful, actual_successful) if successful else []
            sampled_unsuccessful = random.sample(unsuccessful, actual_unsuccessful) if unsuccessful else []
            
            sampled = sampled_successful + sampled_unsuccessful

            question_text = {id:all_question_info[id] for id in sampled}
            
            print(f"  Sampled: {len(sampled)} ({len(sampled_successful)} successful, {len(sampled_unsuccessful)} unsuccessful)")
            
            self.sampled_questions[problem_type] = {
                'problem_ids': sampled,
                'problem_texts': question_text,
                'successful_ids': sampled_successful,
                'unsuccessful_ids': sampled_unsuccessful
            }
    
    def graph_to_prompt(self, graph: nx.DiGraph) -> str:
        """Convert a element graph to a natural language prompt."""
        if graph.number_of_nodes() == 0:
            return "No specific reasoning structure required."
        
        prompt_parts = [
            "Follow this reasoning structure when solving the problem:",
            ""
        ]
        
        # Group elements by their connections
        nodes_info = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            support = node_data.get('support', 0)
            nodes_info.append((node, support))
        
        # Sort by support (most important first)
        nodes_info.sort(key=lambda x: x[1], reverse=True)
        
        # Add nodes with their relationships
        prompt_parts.append("Key reasoning elements to include:")
        for i, (node, support) in enumerate(nodes_info[:10], 1):  # Limit to top 10
            prompt_parts.append(f"{i}. {node}")
        
        # Add edge information for structure
        if graph.number_of_edges() > 0:
            prompt_parts.append("\nReasoning flow:")
            
            # Get edges sorted by support
            edges_info = []
            for u, v, data in graph.edges(data=True):
                edge_type = data.get('edge_type', 'next')
                support = data.get('support', 0)
                edges_info.append((u, v, edge_type, support))
            
            edges_info.sort(key=lambda x: x[3], reverse=True)
            
            for u, v, edge_type, support in edges_info[:15]:  # Limit to top 15
                if edge_type == 'next':
                    prompt_parts.append(f"  - {u} → {v}")
                elif edge_type == 'contains':
                    prompt_parts.append(f"  - {u} (contains) {v}")
                elif edge_type == 'parallel':
                    prompt_parts.append(f"  - {u} (parallel with) {v}")
        
        return "\n".join(prompt_parts)
    
    async def generate_guided_trace(self, question_id, question_info, graph, semaphore, condition_type):
        """Generate a reasoning trace guided by the element graph."""
        async with semaphore:
            # Build the prompt with element guidance
            element_guidance = self.graph_to_prompt(graph)
            
            base_question = question_info['question']
            
            if condition_type == 'guided':
                prompt = f"""{element_guidance}

Question:
{base_question}

Reason through this question following the structure above, then provide your final answer in the following JSON format:
{{
    "final_answer": str
}}
"""
            else:  # baseline - no guidance
                prompt = f"""Reason then answer the following question to the best of your abilities.
Question:
{base_question}

Format your answer in the following JSON format:
{{
    "final_answer": str
}}
"""
            
            try:
                if self.args.no_parser:
                    chat_response = await self.args.client.chat.completions.create(
                        model=self.args.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=self.args.max_tokens,
                        temperature=self.args.temperature,
                        top_p=0.95,
                        extra_body={
                            "top_k": 20,
                            "min_p": 0,
                            "n": 1,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                    )
                    
                    # Parse response
                    output = chat_response.choices[0]
                    
                    # Try to extract reasoning and answer
                    content = output.message.content
                    
                    if "```json" in content:
                        reasoning_content = content.split("```json")[0].strip()
                        try:
                            final_answer = json_repair.loads(content.split("```json")[-1].strip())['final_answer']
                        except:
                            final_answer = content.split("```json")[-1].strip()
                    elif '"final_answer":' in content:
                        reasoning_content = content.split('"final_answer":')[0].strip()
                        try:
                            final_answer = json_repair.loads('{' + content.split('"final_answer":')[1].strip())['final_answer']
                        except:
                            final_answer = content.split('"final_answer":')[1].strip()
                    else:
                        reasoning_content = content
                        final_answer = content
                    
                    if len(reasoning_content) < 3:
                        reasoning_content = ""
                    
                else:
                    chat_response = await self.args.client.chat.completions.create(
                        model=self.args.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "answer",
                                "schema": ReasoningTrace.model_json_schema()
                            },
                        },
                        max_tokens=self.args.max_tokens,
                        temperature=self.args.temperature,
                        top_p=0.95,
                        extra_body={
                            "top_k": 20,
                            "min_p": 0,
                            "n": 1,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                    )
                    
                    output = chat_response.choices[0]
                    reasoning_content = output.message.reasoning_content if hasattr(output.message, 'reasoning_content') else ""
                    final_answer = json_repair.loads(output.message.content)['final_answer']
                
                result = {
                    "reasoning": reasoning_content,
                    "answer": final_answer,
                    "condition": condition_type,
                    "input_tokens": chat_response.usage.prompt_tokens,
                    "output_tokens": chat_response.usage.completion_tokens
                }
                
                return question_id, result
                
            except Exception as e:
                print(f"Error processing question {question_id}: {str(e)}")
                return question_id, None
    
    async def run_guided_generation(self):
        """Run guided generation for all sampled questions."""
        print("\nGenerating guided reasoning traces...")
        
        semaphore = asyncio.Semaphore(self.args.max_concurrent)
        
        results = {}
        
        for problem_type, sample_info in self.sampled_questions.items():
            print(f"\nProcessing problem type: {problem_type}")
            graph = self.problem_type_graphs[problem_type]
            
            results[problem_type] = {
                'questions': {},
                'graph_info': {
                    'num_nodes': graph.number_of_nodes(),
                    'num_edges': graph.number_of_edges()
                }
            }
            
            tasks = []
            for question_id in sample_info['problem_ids']:
                question_info = self.span_tree.problems[question_id]
                
                # Get original question data
                orig_question = {
                    'question': question_info.problem_id,  # You might need to load actual question text
                    'reference_answer': 'N/A',
                    'task': question_info.task_category
                }
                
                # Generate both guided and baseline versions
                tasks.append(
                    self.generate_guided_trace(
                        question_id, orig_question, graph, semaphore, 'guided'
                    )
                )
                tasks.append(
                    self.generate_guided_trace(
                        question_id, orig_question, graph, semaphore, 'baseline'
                    )
                )
            
            # Process with progress bar
            completed_results = []
            for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {problem_type}"):
                result = await coro
                completed_results.append(result)
            
            # Organize results
            for question_id, result in completed_results:
                if result:
                    if question_id not in results[problem_type]['questions']:
                        results[problem_type]['questions'][question_id] = {
                            'problem_id': question_id,
                            'original_correctness': self.span_tree.problems[question_id].correctness.get(self.args.model_name, None),
                            'guided': None,
                            'baseline': None
                        }
                    
                    if result['condition'] == 'guided':
                        results[problem_type]['questions'][question_id]['guided'] = result
                    else:
                        results[problem_type]['questions'][question_id]['baseline'] = result
        
        return results
    
    async def run(self):
        """Main execution flow."""
        # Load data
        self.load_span_tree()
        
        # Build graphs
        self.build_problem_type_graphs()
        
        # Sample questions
        self.sample_questions_per_type()
        
        # Generate guided traces
        results = await self.run_guided_generation()
        
        # Save results
        output_path = os.path.join(
            self.args.output_dir,
            f"{self.args.model_name.split('/')[-1]}_guided_reasoning.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        
        # Print summary statistics
        self.print_summary(results)
    
    def print_summary(self, results):
        """Print summary statistics."""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        for problem_type, data in results.items():
            print(f"\nProblem Type: {problem_type}")
            print(f"  Graph: {data['graph_info']['num_nodes']} nodes, {data['graph_info']['num_edges']} edges")
            print(f"  Questions processed: {len(data['questions'])}")
            
            guided_complete = sum(1 for q in data['questions'].values() if q['guided'] is not None)
            baseline_complete = sum(1 for q in data['questions'].values() if q['baseline'] is not None)
            
            print(f"  Guided completions: {guided_complete}")
            print(f"  Baseline completions: {baseline_complete}")


async def main():
    parser = argparse.ArgumentParser(description="Generate element-guided reasoning traces")
    
    # Model and API settings
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=25000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_concurrent", type=int, default=150)
    parser.add_argument("--no_parser", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")
    
    # Sampling parameters
    parser.add_argument("--samples_per_type", type=int, default=50,
                       help="Number of samples per problem type")
    
    args = parser.parse_args()
    
    # Setup API client
    args.client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.vllm_port}/v1",
        timeout=2400
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run the pipeline
    pipeline = ElementGuidedReasoning(args)
    await pipeline.run()
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    asyncio.run(main())