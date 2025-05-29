#!/usr/bin/env python3
"""
Main entry point for the Novelty Agents System.

This script provides a command-line interface for running the novelty analysis workflow
that integrates evidence from CIVIC, PharmGKB, and Gene Enrichment analyses.
"""

import sys
import os
from pathlib import Path

# Handle imports for both module and direct execution
try:
    from .workflow_runner import NoveltyWorkflowRunner, ValidationError
except ImportError:
    # Fallback for direct execution
    from workflow_runner import NoveltyWorkflowRunner, ValidationError

def main():
    """Main entry point for the novelty analysis workflow."""
    
    def simple_progress_callback(message: str):
        """Simple progress callback for command-line interface."""
        print(f"Progress: {message}")
    
    runner = NoveltyWorkflowRunner(progress_callback=simple_progress_callback)
    
    if len(sys.argv) == 2:
        # Run from JSON config file
        config_file = sys.argv[1]
        print(f"Running novelty analysis from config file: {config_file}")
        
        try:
            result = runner.run_from_json_file(config_file)
            print(f"\n{'='*60}")
            print(f"NOVELTY ANALYSIS WORKFLOW COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Final status: {result.final_status.value}")
            print(f"Total iterations: {result.total_iterations}")
            print(f"Evidence sources integrated:")
            print(f"  • CIVIC genes: {result.consolidated_evidence.total_genes_civic}")
            print(f"  • PharmGKB genes: {result.consolidated_evidence.total_genes_pharmgkb}")
            print(f"  • Gene sets: {result.consolidated_evidence.total_gene_sets_enrichment}")
            
            # Show final report summary
            report = result.unified_report
            print(f"\nFinal Unified Report Sections:")
            sections = [
                ("Potential Novel Biomarkers", len(report.potential_novel_biomarkers)),
                ("Implications", len(report.implications)),
                ("Well-Known Interactions", len(report.well_known_interactions)),
                ("Conclusions", len(report.conclusions))
            ]
            
            for section_name, length in sections:
                status = "✓" if length > 0 else "✗"
                print(f"  {status} {section_name}: {length} characters")
            
            print(f"\nResults saved to: novelty_analysis_consolidated_*.json")
            
        except (ValidationError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)
    
    elif len(sys.argv) == 6:
        # Run with individual file arguments
        civic_file, pharmgkb_file, gene_enrichment_file, context, question = sys.argv[1:6]
        print(f"Running novelty analysis with individual files:")
        print(f"  CIVIC: {civic_file}")
        print(f"  PharmGKB: {pharmgkb_file}")
        print(f"  Gene Enrichment: {gene_enrichment_file}")
        
        try:
            result = runner.run_from_files(
                civic_file=civic_file,
                pharmgkb_file=pharmgkb_file,
                gene_enrichment_file=gene_enrichment_file,
                context=context,
                question=question
            )
            print(f"\n{'='*60}")
            print(f"NOVELTY ANALYSIS WORKFLOW COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Final status: {result.final_status.value}")
            print(f"Total iterations: {result.total_iterations}")
            print(f"Evidence sources integrated:")
            print(f"  • CIVIC genes: {result.consolidated_evidence.total_genes_civic}")
            print(f"  • PharmGKB genes: {result.consolidated_evidence.total_genes_pharmgkb}")
            print(f"  • Gene sets: {result.consolidated_evidence.total_gene_sets_enrichment}")
            
        except (ValidationError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)
    
    else:
        # Show usage and create sample config
        print("Novelty Agents System - Evidence Integration Workflow")
        print("=" * 55)
        print()
        print("This system integrates evidence from three analysis pipelines:")
        print("  • CIVIC Evidence Analysis (clinical interpretations)")
        print("  • PharmGKB Analysis (pharmacogenomic associations)")
        print("  • Gene Enrichment Analysis (pathway and biological processes)")
        print()
        print("The workflow uses 5 specialized agents:")
        print("  1. Orchestrator - coordinates workflow and extracts evidence")
        print("  2. Report Composer - integrates evidence into structured reports")
        print("  3. Evaluator - validates structural integrity and content quality")
        print("  4. Critic - provides critical analysis and identifies biases")
        print("  5. Deliberation - explores counterfactual scenarios and reasoning")
        print()
        
        # Create sample config
        sample_file = runner.create_sample_config()
        
        print("Usage:")
        print(f"  python main.py <config.json>")
        print(f"  python main.py <civic_file> <pharmgkb_file> <gene_enrichment_file> <context> <question>")
        print()
        print("Examples:")
        print(f"  python main.py {sample_file}")
        print(f"  python main.py civic_analysis.json pharmgkb_analysis.json gene_enrichment_analysis.json \"Study context...\" \"Research question?\"")
        print()
        print(f"A sample configuration file has been created: {sample_file}")
        print("Edit this file with your actual file paths and parameters, then run:")
        print(f"  python main.py {sample_file}")

if __name__ == "__main__":
    main()
