from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
import time
import json
from datetime import datetime

from .config import Config
from .models import UserInput, BioExpertOutput, EvaluatorOutput, EvaluationStatus, WorkflowResult, WorkflowMetrics
from .agents import OrchestratorAgent, BioExpertAgent, EvaluatorAgent

class WorkflowEngine:
    """Main workflow engine that orchestrates the biomedical evidence interpretation process."""
    
    def __init__(self, console: Optional[Console] = None, progress_callback=None):
        self.console = console or Console()
        self.progress_callback = progress_callback
        self.orchestrator = OrchestratorAgent()
        self.bioexpert = BioExpertAgent()
        self.evaluator = EvaluatorAgent()
    
    def _update_progress(self, message: str):
        """Update progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(message)
            # Small delay to make progress visible
            time.sleep(0.5)
    
    def _merge_bioexpert_output(self, output: BioExpertOutput) -> str:
        """Merge structured BioExpert output into a single analysis string."""
        sections = []
        
        if output.relevance_explanation:
            sections.append(f"## Relevance Explanation\n{output.relevance_explanation}")
        
        if output.summary_conclusion:
            summary_text = "\n".join(f"- {point}" for point in output.summary_conclusion)
            sections.append(f"## Summary/Conclusion\n{summary_text}")
        
        return "\n\n".join(sections)

    def run_workflow(self, user_input: UserInput) -> WorkflowResult:
        """
        Run the complete workflow for biomedical evidence interpretation.
        
        Args:
            user_input: User's input containing context, question, and evidence
            
        Returns:
            WorkflowResult: Final result with analysis and workflow metadata
        """
        # Extract gene name from evidence string
        if user_input.evidence:
            lines = user_input.evidence.split('\n')
            gene = None
            for line in lines:
                if line.startswith("Gene: "):
                    gene = line.replace("Gene: ", "").strip()
                    break
        else:
            gene = None
        #print("run_workflow user_input:\n", user_input)

        title = f"Starting Biomedical Evidence Interpretation Workflow"
        if gene:
            title += f" for gene {gene}"
            self._update_progress(f"Starting analysis for gene: **{gene}**")
        self.console.print(Panel.fit(title, style="bold blue"))
        
        # Initialize workflow tracking
        evaluation_history: List[EvaluatorOutput] = []
        bioexpert_history: List[BioExpertOutput] = []
        current_bioexpert_output: Optional[BioExpertOutput] = None
        iteration = 1
        workflow_metrics = WorkflowMetrics()
        workflow_start_time = datetime.now()
        
        # Orchestrator starts the workflow
        orchestrator_status = self.orchestrator.coordinate_workflow(user_input)
        self.console.print(f"Orchestrator: {orchestrator_status}\n")
        
        # Main workflow loop
        while iteration <= Config.MAX_ITERATIONS:
            self.console.print(f"**Iteration {iteration}**")
            
            # Update progress for current iteration
            if gene:
                self._update_progress(f"**CIVIC System** - Gene **{gene}** - Iteration **{iteration}**: CIVIC BioExpert analyzing evidence...")
            else:
                self._update_progress(f"**CIVIC System** - Iteration **{iteration}**: CIVIC BioExpert analyzing evidence...")
            
            # BioExpert Analysis Phase
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("CIVIC BioExpert analyzing evidence...", total=None)
                
                # Start tracking BioExpert execution
                bioexpert_execution = self.bioexpert.start_execution()
                
                if iteration == 1:
                    # First analysis
                    current_bioexpert_output = self.bioexpert.analyze(user_input, iteration=iteration)
                else:
                    # Revision based on feedback
                    previous_evaluation = evaluation_history[-1]
                    # Convert feedback_points list to a single feedback string
                    feedback_text = "\n".join(f"- {point}" for point in previous_evaluation.feedback_points)
                    current_bioexpert_output = self.bioexpert.analyze(
                        user_input,
                        previous_output=current_bioexpert_output,
                        evaluator_feedback=feedback_text,
                        iteration=iteration
                    )
                
                # End tracking BioExpert execution
                bioexpert_execution = self.bioexpert.end_execution()
                workflow_metrics.add_agent_execution(bioexpert_execution)
                
                progress.update(task, description="CIVIC BioExpert analysis complete")
                time.sleep(0.5)  # Brief pause for visual effect
            
            # Update progress for evaluation phase
            if gene:
                self._update_progress(f"**CIVIC System** - Gene **{gene}** - Iteration **{iteration}**: CIVIC BioExpert analysis complete, evaluating...")
            else:
                self._update_progress(f"**CIVIC System** - Iteration **{iteration}**: CIVIC BioExpert analysis complete, evaluating...")
            
            # Display BioExpert output
            self._display_bioexpert_output(current_bioexpert_output, iteration)
            
            # Add to history
            bioexpert_history.append(current_bioexpert_output)
            
            # Evaluation phase
            evaluator_execution = self.evaluator.start_execution()
            evaluation = self.evaluator.evaluate(user_input, current_bioexpert_output)
            evaluator_execution = self.evaluator.end_execution()
            workflow_metrics.add_agent_execution(evaluator_execution)
            
            evaluation_history.append(evaluation)
            self._display_evaluation_output(evaluation, iteration)
            
            # Check approval and update progress with status
            if evaluation.status == EvaluationStatus.APPROVED:
                if gene:
                    self._update_progress(f"**CIVIC System** - Gene **{gene}** - Iteration **{iteration}**: Analysis APPROVED by Evaluator!")
                else:
                    self._update_progress(f"**CIVIC System** - Iteration **{iteration}**: Analysis APPROVED by Evaluator!")
                break
            else:
                if gene:
                    self._update_progress(f"**CIVIC System** - Gene **{gene}** - Iteration **{iteration}**: Analysis NOT APPROVED by Evaluator, preparing iteration {iteration + 1}...")
                else:
                    self._update_progress(f"**CIVIC System** - Iteration **{iteration}**: Analysis NOT APPROVED by Evaluator, preparing iteration {iteration + 1}...")
            
            # Prepare next iteration
            iteration += 1
        
        # Final completion message
        if gene:
            self._update_progress(f"**CIVIC System** - Gene **{gene}** analysis completed after **{iteration}** iteration(s)")
        else:
            self._update_progress(f"**CIVIC System** - Analysis completed after **{iteration}** iteration(s)")
        
        # Calculate total workflow time
        workflow_end_time = datetime.now()
        workflow_metrics.total_execution_time_seconds = (workflow_end_time - workflow_start_time).total_seconds()
        
        # Create final result
        merged_analysis = self._merge_bioexpert_output(current_bioexpert_output)
        result = WorkflowResult(
            final_analysis=merged_analysis,
            total_iterations=iteration,
            evaluation_history=evaluation_history,
            bioexpert_history=bioexpert_history,
            final_status=evaluation_history[-1].status,
            user_input=user_input,
            metrics=workflow_metrics
        )
        
        self._display_final_summary(result)
        self._save_result_to_json(result)
        return result

    def _save_result_to_json(self, result: WorkflowResult):
        """Save the workflow result to a JSON file."""
        def default_serializer(obj):
            # Handle enums and objects with __dict__
            if hasattr(obj, "value"):
                return obj.value
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            elif isinstance(obj, set):
                return list(obj)
            return str(obj)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract gene name from evidence string
        if result.user_input.evidence:
            lines = result.user_input.evidence.split('\n')
            gene = "unknown"
            for line in lines:
                if line.startswith("Gene: "):
                    gene = line.replace("Gene: ", "").strip()
                    break
        else:
            gene = "unknown"
        filename = f"workflow_result_{gene}_{timestamp}.json"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result, f, default=default_serializer, indent=2, ensure_ascii=False)
            self.console.print(f"[green]Result saved to {filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to save result: {e}[/red]")

    def _display_bioexpert_output(self, output: BioExpertOutput, iteration: int):
        """Display BioExpert analysis output."""
        panel_title = f"BioExpert Analysis - Iteration {iteration}"
        
        # Use merged analysis for display
        display_analysis = self._merge_bioexpert_output(output)
        if len(display_analysis) > 2500:
            display_analysis = display_analysis[:2500] + "\n\n[... truncated for display ...]"
        
        content = f"**Analysis:**\n{display_analysis}"
        
        self.console.print(Panel(content, title=panel_title, style="cyan"))
    
    def _display_evaluation_output(self, evaluation: EvaluatorOutput, iteration: int):
        """Display Evaluator output."""
        if evaluation.status == EvaluationStatus.APPROVED:
            panel_title = "Evaluation Result - APPROVED"
            style = "green"
            feedback_text = "\n".join(f"- {point}" for point in evaluation.feedback_points) if evaluation.feedback_points else "No additional feedback provided."
            content = f"**Status:** APPROVED\n\n**Feedback:**\n{feedback_text}"
        else:
            panel_title = "Evaluation Result - NOT APPROVED"
            style = "red"
            feedback_text = "\n".join(f"- {point}" for point in evaluation.feedback_points) if evaluation.feedback_points else "No specific feedback provided."
            content = f"""**Status:** NOT APPROVED

**Feedback:**
{feedback_text}

**Issues Identified:** {len(evaluation.feedback_points or [])}
"""
        
        self.console.print(Panel(content, title=panel_title, style=style))
        print()  # Add spacing
    
    def _display_final_summary(self, result: WorkflowResult):
        """Display final workflow summary."""
        status_emoji = "" if result.final_status == EvaluationStatus.APPROVED else ""
        status_color = "green" if result.final_status == EvaluationStatus.APPROVED else "yellow"
        
        # Extract gene name from evidence string
        gene = None
        if result.user_input.evidence:
            lines = result.user_input.evidence.split('\n')
            for line in lines:
                if line.startswith("Gene: "):
                    gene = line.replace("Gene: ", "").strip()
                    break
        
        summary = f"""
                    **Final Status:** {status_emoji} {result.final_status.value}
                    **Gene:** {gene if gene else 'N/A'}
                    **Total Iterations:** {result.total_iterations}
                    **Question:** {result.user_input.question}
                    **Evidence Length:** {len(result.user_input.evidence)} characters

                    **Workflow Complete!**
                    """
        
        self.console.print(Panel(summary, title="Workflow Summary", style=status_color))

def create_workflow_engine(progress_callback=None) -> WorkflowEngine:
    """Factory function to create a workflow engine."""
    return WorkflowEngine(progress_callback=progress_callback)
