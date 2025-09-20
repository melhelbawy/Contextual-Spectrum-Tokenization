"""
Evaluation Framework for CST Models
Comprehensive benchmarking suite for semantic disambiguation, efficiency, and multimodal tasks
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr

from cst_transformer import CSTransformer
from config import CSTConfig


logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiling utilities"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.counters = defaultdict(int)
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        return self.TimingContext(self, operation_name)
    
    class TimingContext:
        def __init__(self, profiler, operation_name):
            self.profiler = profiler
            self.operation_name = operation_name
            self.start_time = None
            self.start_memory = None
        
        def __enter__(self):
            self.start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.start_memory = torch.cuda.memory_allocated()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                memory_diff = end_memory - (self.start_memory or 0)
                self.profiler.memory_usage[self.operation_name].append(memory_diff)
            
            end_time = time.time()
            elapsed = end_time - self.start_time
            self.profiler.timings[self.operation_name].append(elapsed)
            self.profiler.counters[self.operation_name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for op_name, times in self.timings.items():
            stats[f"{op_name}_timing"] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'p50': np.percentile(times, 50),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99),
                'total': np.sum(times),
                'count': len(times)
            }
        
        for op_name, memory in self.memory_usage.items():
            if memory:
                stats[f"{op_name}_memory"] = {
                    'mean_mb': np.mean(memory) / 1024 / 1024,
                    'max_mb': np.max(memory) / 1024 / 1024,
                    'total_mb': np.sum(memory) / 1024 / 1024
                }
        
        return stats


class WordSenseDisambiguationEvaluator:
    """Evaluator for Word Sense Disambiguation tasks"""
    
    def __init__(self, model: CSTransformer, config: CSTConfig):
        self.model = model
        self.config = config
        self.profiler = PerformanceProfiler()
    
    def evaluate_on_semeval(self, dataset_path: str) -> Dict[str, Any]:
        """Evaluate on SemEval WSD datasets"""
        # Load SemEval data (simplified - adapt to actual format)
        test_data = self._load_semeval_data(dataset_path)
        
        results = {
            'predictions': [],
            'ground_truth': [],
            'ambiguous_words': [],
            'context_lengths': [],
            'prediction_confidences': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for item in test_data:
                with self.profiler.time_operation('wsd_inference'):
                    prediction, confidence = self._predict_word_sense(
                        item['sentence'],
                        item['target_word'],
                        item['target_position'],
                        item['sense_candidates']
                    )
                
                results['predictions'].append(prediction)
                results['ground_truth'].append(item['correct_sense'])
                results['ambiguous_words'].append(item['target_word'])
                results['context_lengths'].append(len(item['sentence'].split()))
                results['prediction_confidences'].append(confidence)
        
        # Compute metrics
        accuracy = accuracy_score(results['ground_truth'], results['predictions'])
        f1 = f1_score(results['ground_truth'], results['predictions'], average='weighted')
        
        # Per-word analysis
        word_accuracies = self._compute_per_word_accuracy(
            results['ambiguous_words'], 
            results['ground_truth'], 
            results['predictions']
        )
        
        # Context length analysis
        context_analysis = self._analyze_by_context_length(
            results['context_lengths'],
            results['ground_truth'],
            results['predictions']
        )
        
        return {
            'overall_accuracy': accuracy,
            'weighted_f1': f1,
            'per_word_accuracy': word_accuracies,
            'context_length_analysis': context_analysis,
            'performance_stats': self.profiler.get_stats(),
            'num_samples': len(test_data)
        }
    
    def _load_semeval_data(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load SemEval WSD data - simplified implementation"""
        # This is a placeholder - implement based on actual SemEval format
        synthetic_data = []
        
        ambiguous_words = ['bank', 'plant', 'scale', 'rock', 'bark', 'crown', 'mouse', 'bat']
        senses = {
            'bank': ['financial_institution', 'river_side'],
            'plant': ['factory', 'vegetation'],
            'scale': ['measurement', 'fish_covering'],
            'rock': ['stone', 'music_genre'],
            'bark': ['dog_sound', 'tree_covering'],
            'crown': ['royal_headwear', 'tooth_covering'],
            'mouse': ['computer_device', 'animal'],
            'bat': ['sports_equipment', 'flying_mammal']
        }
        
        for i in range(200):  # Generate 200 test cases
            word = np.random.choice(ambiguous_words)
            sense_candidates = senses[word]
            correct_sense = np.random.choice(sense_candidates)
            
            # Generate context sentence
            if word == 'bank':
                if correct_sense == 'financial_institution':
                    sentence = f"I went to the {word} to deposit money and check my account balance."
                else:
                    sentence = f"We sat by the {word} of the river watching the sunset."
            elif word == 'plant':
                if correct_sense == 'factory':
                    sentence = f"The manufacturing {word} operates 24 hours a day."
                else:
                    sentence = f"This {word} needs water and sunlight to grow properly."
            else:
                sentence = f"The {word} is important in this context for disambiguation."
            
            target_pos = sentence.split().index(word)
            
            synthetic_data.append({
                'sentence': sentence,
                'target_word': word,
                'target_position': target_pos,
                'sense_candidates': sense_candidates,
                'correct_sense': correct_sense
            })
        
        return synthetic_data
    
    def _predict_word_sense(self, sentence: str, target_word: str, 
                          target_position: int, sense_candidates: List[str]) -> Tuple[str, float]:
        """Predict word sense using CST model"""
        # Tokenize sentence (simplified)
        words = sentence.split()
        
        # Create input for model
        input_ids = torch.tensor([[hash(w) % self.config.vocab_size for w in words]], dtype=torch.long)
        
        # Create context data emphasizing the target word
        context_data = {
            'document_embedding': torch.randn(1, self.config.raw_doc_dim),
            'metadata': {
                'author': torch.tensor([0]),
                'domain': torch.tensor([0]),
                'timestamp': torch.tensor([0.0])
            }
        }
        
        # Get model outputs
        outputs = self.model(input_ids, context_data)
        
        # Extract representation for target word
        target_repr = outputs['hidden_states'][0, target_position]  # [d_model]
        
        # Compute similarity with sense embeddings (simplified)
        sense_scores = []
        for sense in sense_candidates:
            # Generate sense embedding (in practice, use pre-trained sense embeddings)
            sense_embedding = torch.randn(self.config.d_model)
            similarity = F.cosine_similarity(target_repr, sense_embedding, dim=0)
            sense_scores.append(similarity.item())
        
        # Predict sense with highest similarity
        best_sense_idx = np.argmax(sense_scores)
        confidence = torch.softmax(torch.tensor(sense_scores), dim=0)[best_sense_idx].item()
        
        return sense_candidates[best_sense_idx], confidence
    
    def _compute_per_word_accuracy(self, words: List[str], 
                                 ground_truth: List[str], 
                                 predictions: List[str]) -> Dict[str, float]:
        """Compute accuracy for each ambiguous word"""
        word_results = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for word, gt, pred in zip(words, ground_truth, predictions):
            word_results[word]['total'] += 1
            if gt == pred:
                word_results[word]['correct'] += 1
        
        return {word: stats['correct'] / stats['total'] 
                for word, stats in word_results.items()}
    
    def _analyze_by_context_length(self, context_lengths: List[int],
                                 ground_truth: List[str],
                                 predictions: List[str]) -> Dict[str, float]:
        """Analyze performance by context length"""
        length_buckets = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]
        bucket_results = {}
        
        for min_len, max_len in length_buckets:
            mask = [(min_len <= length < max_len) for length in context_lengths]
            if not any(mask):
                continue
                
            bucket_gt = [gt for gt, m in zip(ground_truth, mask) if m]
            bucket_pred = [pred for pred, m in zip(predictions, mask) if m]
            
            if bucket_gt:
                accuracy = accuracy_score(bucket_gt, bucket_pred)
                bucket_name = f"{min_len}-{max_len if max_len != float('inf') else '∞'}"
                bucket_results[bucket_name] = accuracy
        
        return bucket_results


class EfficiencyEvaluator:
    """Evaluator for computational efficiency"""
    
    def __init__(self, cst_model: CSTransformer, baseline_models: Dict[str, Any]):
        self.cst_model = cst_model
        self.baseline_models = baseline_models
        self.profiler = PerformanceProfiler()
    
    def benchmark_inference_speed(self, test_sequences: List[torch.Tensor],
                                context_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark inference speed comparison"""
        
        results = {
            'cst_model': {'times': [], 'memory': []},
            'baselines': {name: {'times': [], 'memory': []} for name in self.baseline_models}
        }
        
        # Benchmark CST model
        self.cst_model.eval()
        with torch.no_grad():
            for seq, context_data in zip(test_sequences, context_data_list):
                with self.profiler.time_operation('cst_inference'):
                    _ = self.cst_model(seq.unsqueeze(0), context_data)
        
        cst_stats = self.profiler.get_stats()
        results['cst_model'] = cst_stats.get('cst_inference_timing', {})
        
        # Benchmark baseline models
        for name, baseline_model in self.baseline_models.items():
            baseline_model.eval()
            profiler = PerformanceProfiler()
            
            with torch.no_grad():
                for seq in test_sequences:
                    with profiler.time_operation('baseline_inference'):
                        if hasattr(baseline_model, 'forward'):
                            _ = baseline_model(seq.unsqueeze(0))
                        else:
                            # Handle different baseline interfaces
                            _ = baseline_model(seq.unsqueeze(0))
            
            baseline_stats = profiler.get_stats()
            results['baselines'][name] = baseline_stats.get('baseline_inference_timing', {})
        
        # Compute relative performance
        cst_mean_time = results['cst_model'].get('mean', 0)
        relative_performance = {}
        
        for name, stats in results['baselines'].items():
            baseline_mean_time = stats.get('mean', 0)
            if baseline_mean_time > 0:
                relative_performance[name] = cst_mean_time / baseline_mean_time
        
        results['relative_performance'] = relative_performance
        
        return results
    
    def analyze_cache_performance(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze CST cache performance"""
        
        # Clear cache and enable profiling
        self.cst_model.cst_module.clear_cache()
        self.cst_model.enable_cst_profiling(True)
        
        cache_stats_over_time = []
        
        self.cst_model.eval()
        with torch.no_grad():
            for i, item in enumerate(test_data):
                _ = self.cst_model(
                    item['input_ids'].unsqueeze(0),
                    item['context_data']
                )
                
                if i % 100 == 0:  # Sample every 100 steps
                    stats = self.cst_model.get_cst_stats()
                    cache_stats_over_time.append({
                        'step': i,
                        'hit_rate': stats.get('hit_rate', 0),
                        'cache_size': stats.get('cache_size', 0),
                        'ambiguous_ratio': stats.get('ambiguous_ratio', 0)
                    })
        
        final_stats = self.cst_model.get_cst_stats()
        
        return {
            'final_cache_stats': final_stats,
            'cache_evolution': cache_stats_over_time,
            'cache_efficiency_analysis': self._analyze_cache_efficiency(cache_stats_over_time)
        }
    
    def _analyze_cache_efficiency(self, cache_evolution: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cache efficiency over time"""
        if not cache_evolution:
            return {}
        
        hit_rates = [stats['hit_rate'] for stats in cache_evolution]
        cache_sizes = [stats['cache_size'] for stats in cache_evolution]
        
        return {
            'hit_rate_trend': {
                'initial': hit_rates[0] if hit_rates else 0,
                'final': hit_rates[-1] if hit_rates else 0,
                'peak': max(hit_rates) if hit_rates else 0,
                'mean': np.mean(hit_rates) if hit_rates else 0
            },
            'cache_utilization': {
                'mean_size': np.mean(cache_sizes) if cache_sizes else 0,
                'max_size': max(cache_sizes) if cache_sizes else 0,
                'final_size': cache_sizes[-1] if cache_sizes else 0
            }
        }


class MultimodalEvaluator:
    """Evaluator for multimodal understanding tasks"""
    
    def __init__(self, model: CSTransformer, config: CSTConfig):
        self.model = model
        self.config = config
        self.profiler = PerformanceProfiler()
    
    def evaluate_visual_question_answering(self, vqa_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate on Visual Question Answering tasks"""
        
        results = {
            'predictions': [],
            'ground_truth': [],
            'question_types': [],
            'confidence_scores': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for item in vqa_dataset:
                with self.profiler.time_operation('vqa_inference'):
                    prediction, confidence = self._answer_visual_question(
                        item['question'],
                        item['image_features'],
                        item['answer_candidates']
                    )
                
                results['predictions'].append(prediction)
                results['ground_truth'].append(item['correct_answer'])
                results['question_types'].append(item.get('question_type', 'unknown'))
                results['confidence_scores'].append(confidence)
        
        # Compute metrics
        accuracy = accuracy_score(results['ground_truth'], results['predictions'])
        
        # Per question type analysis
        type_analysis = self._analyze_by_question_type(
            results['question_types'],
            results['ground_truth'],
            results['predictions']
        )
        
        return {
            'overall_accuracy': accuracy,
            'per_question_type': type_analysis,
            'mean_confidence': np.mean(results['confidence_scores']),
            'performance_stats': self.profiler.get_stats()
        }
    
    def _answer_visual_question(self, question: str, image_features: torch.Tensor,
                              answer_candidates: List[str]) -> Tuple[str, float]:
        """Answer a visual question using CST model"""
        
        # Tokenize question
        question_words = question.split()
        input_ids = torch.tensor([[hash(w) % self.config.vocab_size for w in question_words]], dtype=torch.long)
        
        # Create multimodal context
        context_data = {
            'document_embedding': torch.randn(1, self.config.raw_doc_dim),
            'metadata': {
                'author': torch.tensor([0]),
                'domain': torch.tensor([0]),  # VQA domain
                'timestamp': torch.tensor([0.0])
            },
            'multimodal': {
                'image_clip': image_features.unsqueeze(0)  # [1, clip_dim]
            }
        }
        
        # Get model representation
        outputs = self.model(input_ids, context_data)
        question_repr = outputs['hidden_states'].mean(dim=1)  # Pool over sequence
        
        # Compare with answer embeddings (simplified)
        answer_scores = []
        for answer in answer_candidates:
            answer_embedding = torch.randn(self.config.d_model)  # Placeholder
            similarity = F.cosine_similarity(question_repr.squeeze(), answer_embedding, dim=0)
            answer_scores.append(similarity.item())
        
        best_answer_idx = np.argmax(answer_scores)
        confidence = torch.softmax(torch.tensor(answer_scores), dim=0)[best_answer_idx].item()
        
        return answer_candidates[best_answer_idx], confidence
    
    def _analyze_by_question_type(self, question_types: List[str],
                                ground_truth: List[str],
                                predictions: List[str]) -> Dict[str, float]:
        """Analyze VQA performance by question type"""
        type_results = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for qtype, gt, pred in zip(question_types, ground_truth, predictions):
            type_results[qtype]['total'] += 1
            if gt == pred:
                type_results[qtype]['correct'] += 1
        
        return {qtype: stats['correct'] / stats['total'] 
                for qtype, stats in type_results.items()}


class ComprehensiveEvaluator:
    """Main evaluator that orchestrates all evaluation tasks"""
    
    def __init__(self, cst_model: CSTransformer, baseline_models: Dict[str, Any], config: CSTConfig):
        self.cst_model = cst_model
        self.baseline_models = baseline_models
        self.config = config
        
        # Initialize sub-evaluators
        self.wsd_evaluator = WordSenseDisambiguationEvaluator(cst_model, config)
        self.efficiency_evaluator = EfficiencyEvaluator(cst_model, baseline_models)
        self.multimodal_evaluator = MultimodalEvaluator(cst_model, config)
    
    def run_full_evaluation(self, test_datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive evaluation across all tasks"""
        
        results = {}
        
        # 1. Word Sense Disambiguation
        if 'wsd' in test_datasets:
            logger.info("Running Word Sense Disambiguation evaluation...")
            wsd_results = self.wsd_evaluator.evaluate_on_semeval(test_datasets['wsd'])
            results['word_sense_disambiguation'] = wsd_results
        
        # 2. Efficiency Benchmarking
        if 'efficiency' in test_datasets:
            logger.info("Running efficiency benchmarks...")
            efficiency_results = self.efficiency_evaluator.benchmark_inference_speed(
                test_datasets['efficiency']['sequences'],
                test_datasets['efficiency']['context_data']
            )
            results['efficiency'] = efficiency_results
        
        # 3. Cache Performance Analysis
        if 'cache_test' in test_datasets:
            logger.info("Analyzing cache performance...")
            cache_results = self.efficiency_evaluator.analyze_cache_performance(
                test_datasets['cache_test']
            )
            results['cache_performance'] = cache_results
        
        # 4. Multimodal Tasks
        if 'vqa' in test_datasets:
            logger.info("Running Visual Question Answering evaluation...")
            vqa_results = self.multimodal_evaluator.evaluate_visual_question_answering(
                test_datasets['vqa']
            )
            results['visual_question_answering'] = vqa_results
        
        # 5. GLUE-style benchmarks
        if 'glue' in test_datasets:
            logger.info("Running GLUE benchmark tasks...")
            glue_results = self.evaluate_glue_tasks(test_datasets['glue'])
            results['glue_benchmark'] = glue_results
        
        # 6. Generate comprehensive report
        report = self.generate_evaluation_report(results)
        results['comprehensive_report'] = report
        
        return results
    
    def evaluate_glue_tasks(self, glue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate on GLUE-style tasks"""
        
        glue_results = {}
        
        for task_name, task_data in glue_data.items():
            logger.info(f"Evaluating on {task_name}...")
            
            # Create task-specific model if needed
            task_model = CSTransformer(self.config, task_type='classification')
            task_model.load_state_dict(self.cst_model.state_dict(), strict=False)
            
            predictions = []
            ground_truth = []
            
            task_model.eval()
            with torch.no_grad():
                for item in task_data:
                    # Prepare input
                    input_ids = torch.tensor([item['input_ids']], dtype=torch.long)
                    context_data = item['context_data']
                    
                    # Forward pass
                    outputs = task_model(input_ids, context_data)
                    
                    # Get prediction
                    logits = outputs['logits']
                    prediction = torch.argmax(logits, dim=-1).item()
                    
                    predictions.append(prediction)
                    ground_truth.append(item['label'])
            
            # Compute metrics
            accuracy = accuracy_score(ground_truth, predictions)
            f1 = f1_score(ground_truth, predictions, average='weighted')
            
            glue_results[task_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'num_samples': len(task_data)
            }
        
        return glue_results
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        report = {
            'summary': {},
            'detailed_analysis': {},
            'comparisons': {},
            'recommendations': []
        }
        
        # Summary metrics
        if 'word_sense_disambiguation' in results:
            wsd_acc = results['word_sense_disambiguation']['overall_accuracy']
            report['summary']['wsd_accuracy'] = wsd_acc
            
            if wsd_acc > 0.8:
                report['recommendations'].append("Excellent WSD performance - suitable for disambiguation tasks")
            elif wsd_acc > 0.6:
                report['recommendations'].append("Good WSD performance - consider fine-tuning for critical applications")
            else:
                report['recommendations'].append("WSD performance needs improvement - review training data and context features")
        
        # Efficiency analysis
        if 'efficiency' in results:
            rel_perf = results['efficiency'].get('relative_performance', {})
            report['summary']['efficiency_vs_baselines'] = rel_perf
            
            avg_slowdown = np.mean(list(rel_perf.values())) if rel_perf else 1.0
            if avg_slowdown < 1.5:
                report['recommendations'].append("Efficient inference - suitable for production deployment")
            elif avg_slowdown < 3.0:
                report['recommendations'].append("Moderate overhead - optimize caching for better performance")
            else:
                report['recommendations'].append("High computational overhead - consider model compression")
        
        # Cache effectiveness
        if 'cache_performance' in results:
            final_hit_rate = results['cache_performance']['final_cache_stats'].get('hit_rate', 0)
            report['summary']['cache_hit_rate'] = final_hit_rate
            
            if final_hit_rate > 0.7:
                report['recommendations'].append("Excellent cache performance - dynamic processing is well-optimized")
            elif final_hit_rate > 0.4:
                report['recommendations'].append("Good cache performance - consider increasing cache size")
            else:
                report['recommendations'].append("Low cache hit rate - review ambiguity detection or increase cache capacity")
        
        # Multimodal performance
        if 'visual_question_answering' in results:
            vqa_acc = results['visual_question_answering']['overall_accuracy']
            report['summary']['vqa_accuracy'] = vqa_acc
            
            if vqa_acc > 0.6:
                report['recommendations'].append("Strong multimodal understanding - CST effectively integrates visual information")
            else:
                report['recommendations'].append("Multimodal performance needs improvement - enhance image processing pipeline")
        
        # Overall assessment
        accuracies = []
        if 'wsd_accuracy' in report['summary']:
            accuracies.append(report['summary']['wsd_accuracy'])
        if 'vqa_accuracy' in report['summary']:
            accuracies.append(report['summary']['vqa_accuracy'])
        
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            report['summary']['overall_performance'] = avg_accuracy
            
            if avg_accuracy > 0.75:
                report['summary']['assessment'] = "Excellent overall performance"
            elif avg_accuracy > 0.6:
                report['summary']['assessment'] = "Good overall performance"
            else:
                report['summary']['assessment'] = "Performance needs improvement"
        
        return report
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        
        # Convert tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_tensors(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def plot_performance_comparison(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Generate performance comparison plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. WSD accuracy by word
        if 'word_sense_disambiguation' in results:
            wsd_data = results['word_sense_disambiguation']['per_word_accuracy']
            words = list(wsd_data.keys())
            accuracies = list(wsd_data.values())
            
            axes[0, 0].bar(words, accuracies)
            axes[0, 0].set_title('WSD Accuracy by Word')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Efficiency comparison
        if 'efficiency' in results:
            rel_perf = results['efficiency']['relative_performance']
            models = list(rel_perf.keys())
            speedups = [1/perf for perf in rel_perf.values()]  # Convert to speedup
            
            axes[0, 1].bar(models, speedups)
            axes[0, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Inference Speed Comparison (Speedup vs CST)')
            axes[0, 1].set_ylabel('Speedup Factor')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Cache performance over time
        if 'cache_performance' in results:
            cache_evolution = results['cache_performance']['cache_evolution']
            steps = [item['step'] for item in cache_evolution]
            hit_rates = [item['hit_rate'] for item in cache_evolution]
            
            axes[1, 0].plot(steps, hit_rates, 'b-', linewidth=2)
            axes[1, 0].set_title('Cache Hit Rate Over Time')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Hit Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Overall performance summary
        summary_metrics = []
        summary_values = []
        
        if 'word_sense_disambiguation' in results:
            summary_metrics.append('WSD')
            summary_values.append(results['word_sense_disambiguation']['overall_accuracy'])
        
        if 'visual_question_answering' in results:
            summary_metrics.append('VQA')
            summary_values.append(results['visual_question_answering']['overall_accuracy'])
        
        if 'glue_benchmark' in results:
            glue_scores = [task['accuracy'] for task in results['glue_benchmark'].values()]
            summary_metrics.append('GLUE Avg')
            summary_values.append(np.mean(glue_scores))
        
        if summary_metrics:
            axes[1, 1].bar(summary_metrics, summary_values)
            axes[1, 1].set_title('Overall Performance Summary')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {save_path}")
        
        plt.show()


def create_test_datasets(config: CSTConfig) -> Dict[str, Any]:
    """Create synthetic test datasets for evaluation"""
    
    datasets = {}
    
    # WSD dataset
    datasets['wsd'] = "synthetic_wsd_data"  # Path placeholder
    
    # Efficiency test data
    test_sequences = []
    context_data_list = []
    
    for i in range(100):
        seq_len = np.random.randint(10, 100)
        seq = torch.randint(1, config.vocab_size, (seq_len,))
        
        context_data = {
            'document_embedding': torch.randn(config.raw_doc_dim),
            'metadata': {
                'author': torch.randint(0, config.num_authors, (1,)).item(),
                'domain': torch.randint(0, config.num_domains, (1,)).item(),
                'timestamp': torch.randn(1).item(),
            }
        }
        
        test_sequences.append(seq)
        context_data_list.append(context_data)
    
    datasets['efficiency'] = {
        'sequences': test_sequences,
        'context_data': context_data_list
    }
    
    # Cache test data
    cache_test_data = []
    for i in range(1000):
        seq_len = np.random.randint(5, 50)
        input_ids = torch.randint(1, config.vocab_size, (seq_len,))
        
        context_data = {
            'document_embedding': torch.randn(config.raw_doc_dim),
            'metadata': {
                'author': torch.randint(0, 10, (1,)).item(),  # Limited authors for cache hits
                'domain': torch.randint(0, 5, (1,)).item(),   # Limited domains for cache hits
                'timestamp': torch.randn(1).item(),
            }
        }
        
        cache_test_data.append({
            'input_ids': input_ids,
            'context_data': context_data
        })
    
    datasets['cache_test'] = cache_test_data
    
    # VQA dataset
    vqa_data = []
    for i in range(200):
        questions = [
            "What color is the object?",
            "How many items are visible?",
            "What is the person doing?",
            "Where is this photo taken?",
            "What type of animal is shown?"
        ]
        
        question = np.random.choice(questions)
        image_features = torch.randn(config.clip_dim)
        
        if "color" in question:
            answer_candidates = ["red", "blue", "green", "yellow", "black"]
        elif "many" in question:
            answer_candidates = ["one", "two", "three", "four", "many"]
        elif "doing" in question:
            answer_candidates = ["walking", "running", "sitting", "standing", "eating"]
        elif "where" in question:
            answer_candidates = ["park", "home", "office", "street", "beach"]
        else:
            answer_candidates = ["dog", "cat", "bird", "horse", "elephant"]
        
        vqa_data.append({
            'question': question,
            'image_features': image_features,
            'answer_candidates': answer_candidates,
            'correct_answer': np.random.choice(answer_candidates),
            'question_type': question.split()[0].lower()
        })
    
    datasets['vqa'] = vqa_data
    
    return datasets


def main():
    """Main evaluation script"""
    
    # Setup
    config = CSTConfig()
    config.ambiguous_word_ids = [1, 5, 10, 15, 20, 25, 30]
    
    # Load models
    cst_model = CSTransformer(config, task_type='mlm')
    
    # Create dummy baseline models for comparison
    baseline_models = {
        'standard_bert': torch.nn.Sequential(
            torch.nn.Embedding(config.vocab_size, config.d_model),
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(config.d_model, 8),
                num_layers=6
            )
        )
    }
    
    # Create test datasets
    test_datasets = create_test_datasets(config)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(cst_model, baseline_models, config)
    
    # Run evaluation
    logger.info("Starting comprehensive evaluation...")
    results = evaluator.run_full_evaluation(test_datasets)
    
    # Save results
    evaluator.save_results(results, 'cst_evaluation_results.json')
    
    # Generate plots
    evaluator.plot_performance_comparison(results, 'cst_performance_plots.png')
    
    # Print summary
    if 'comprehensive_report' in results:
        report = results['comprehensive_report']
        print("\n" + "="*50)
        print("CST EVALUATION SUMMARY")
        print("="*50)
        print(f"Overall Assessment: {report['summary'].get('assessment', 'N/A')}")
        print(f"Overall Performance: {report['summary'].get('overall_performance', 0):.3f}")
        print(f"WSD Accuracy: {report['summary'].get('wsd_accuracy', 0):.3f}")
        print(f"Cache Hit Rate: {report['summary'].get('cache_hit_rate', 0):.3f}")
        print(f"VQA Accuracy: {report['summary'].get('vqa_accuracy', 0):.3f}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        print("="*50)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()