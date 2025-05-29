"""
PDF Report Generation Module for Molecular Interaction Signatures Portal

This module provides functionality to generate professional PDF reports 
for Evidence Integration analysis results.

Author: MIS Portal Team
"""

from datetime import datetime
import io


def generate_evidence_integration_pdf(context, question, gene_list, evidence_integration_data, analysis_timestamp):
    """
    Generate a PDF report for Evidence Integration analysis
    
    Args:
        context: User context
        question: User question/prompt  
        gene_list: List of genes analyzed
        evidence_integration_data: Evidence integration consolidated data
        analysis_timestamp: When the analysis was performed
    
    Returns:
        bytes: PDF content as bytes, or None if reportlab is not available
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.lib.colors import HexColor
        
        # Create a buffer to store PDF content
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                               leftMargin=0.75*inch, rightMargin=0.75*inch,
                               topMargin=1*inch, bottomMargin=1*inch)
        
        # Build story content
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=HexColor('#1f77b4')
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=HexColor('#2c3e50')
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        # Title
        story.append(Paragraph("Molecular Interaction Signatures Portal", title_style))
        story.append(Paragraph("AI-Powered Evidence Integration Report", subtitle_style))
        story.append(Spacer(1, 20))
        
        # Analysis Information
        story.append(Paragraph("Analysis Information", subtitle_style))
        
        info_data = [
            ["Report Generated:", analysis_timestamp.strftime("%B %d, %Y at %H:%M:%S")],
            ["Analysis Type:", "Evidence Integration"],
            ["Total Genes Analyzed:", str(len(gene_list)) if gene_list else "N/A"],
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#f8f9fa')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Context Section
        if context and context.strip():
            story.append(Paragraph("Research Context", subtitle_style))
            story.append(Paragraph(context.strip(), body_style))
            story.append(Spacer(1, 15))
        
        # Question Section
        if question and question.strip():
            story.append(Paragraph("Research Question", subtitle_style))
            story.append(Paragraph(question.strip(), body_style))
            story.append(Spacer(1, 15))
        
        # Gene List Section
        if gene_list:
            story.append(Paragraph("Genes Analyzed", subtitle_style))
            
            # Format gene list nicely
            if len(gene_list) <= 20:
                gene_text = ", ".join(gene_list)
            else:
                gene_text = ", ".join(gene_list[:20]) + f"... and {len(gene_list) - 20} more genes"
            
            story.append(Paragraph(gene_text, body_style))
            story.append(Spacer(1, 15))
        
        # Evidence Integration Results
        if evidence_integration_data and hasattr(evidence_integration_data, 'unified_report') and evidence_integration_data.unified_report:
            story.append(Paragraph("AI Analysis Results", subtitle_style))
            
            unified_report = evidence_integration_data.unified_report
            
            # Helper function to clean and format text for PDF
            def clean_text_for_pdf(text):
                if not text:
                    return "No information available."
                
                # Remove markdown formatting and clean up text
                import re
                
                # Remove markdown headers
                text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
                
                # Convert bullet points
                text = re.sub(r'^[•\-\*]\s*', '• ', text, flags=re.MULTILINE)
                
                # Clean up extra whitespace
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = text.strip()
                
                return text
            
            # Potential Novel Biomarkers
            if unified_report.potential_novel_biomarkers:
                story.append(Paragraph("Potential Novel Biomarkers", styles['Heading3']))
                cleaned_text = clean_text_for_pdf(unified_report.potential_novel_biomarkers)
                story.append(Paragraph(cleaned_text, body_style))
                story.append(Spacer(1, 12))
            
            # Implications
            if unified_report.implications:
                story.append(Paragraph("Clinical and Research Implications", styles['Heading3']))
                cleaned_text = clean_text_for_pdf(unified_report.implications)
                story.append(Paragraph(cleaned_text, body_style))
                story.append(Spacer(1, 12))
            
            # Well-Known Interactions
            if unified_report.well_known_interactions:
                story.append(Paragraph("Well-Known Interactions", styles['Heading3']))
                cleaned_text = clean_text_for_pdf(unified_report.well_known_interactions)
                story.append(Paragraph(cleaned_text, body_style))
                story.append(Spacer(1, 12))
            
            # Conclusions
            if unified_report.conclusions:
                story.append(Paragraph("Conclusions", styles['Heading3']))
                cleaned_text = clean_text_for_pdf(unified_report.conclusions)
                story.append(Paragraph(cleaned_text, body_style))
                story.append(Spacer(1, 12))
        
        # Analysis Summary
        if evidence_integration_data:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Analysis Summary", subtitle_style))
            
            summary_data = [
                ["Total Iterations:", str(evidence_integration_data.total_iterations)],
                ["Analysis Status:", str(evidence_integration_data.final_status.value) if hasattr(evidence_integration_data.final_status, 'value') else str(evidence_integration_data.final_status)],
            ]
            
            if hasattr(evidence_integration_data, 'consolidated_evidence'):
                ce = evidence_integration_data.consolidated_evidence
                summary_data.extend([
                    ["CIVIC Genes:", str(ce.total_genes_civic)],
                    ["PharmGKB Genes:", str(ce.total_genes_pharmgkb)], 
                    ["Gene Sets:", str(ce.total_gene_sets_enrichment)],
                ])
            
            summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), HexColor('#f8f9fa')),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(summary_table)
        
        # Footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=HexColor('#6c757d')
        )
        story.append(Paragraph("Generated by Molecular Interaction Signatures Portal - AI Assistant", footer_style))
        story.append(Paragraph("This report is generated for research purposes only.", footer_style))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
        
    except ImportError:
        # If reportlab is not available, return None
        return None
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None 