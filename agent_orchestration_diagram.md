# Agent Systems Orchestration Diagram

This diagram shows how the four agent systems work together in the CEE DART Navigator.

```mermaid
flowchart TD
    %% Input Data
    Input[("📊 Input Data<br/>• Context<br/>• Question<br/>• Pre-extracted Evidence<br/>from Knowledge Bases")]
    
    %% Evidence Collection Phase
    subgraph Evidence["🔍 Evidence Analysis Phase (Parallel Execution)"]
        direction TB
        
        %% CIVIC System
        subgraph CIVIC["🧬 CIVIC Analysis System"]
            direction TB
            C_Bio["BioExpert Agent<br/>(Clinical Evidence Analysis)"]
            C_Eval["Content Validator<br/>(Quality Check)"]
            C_Bio --> C_Eval
            C_Eval -->|"APPROVED/NOT APPROVED"| C_Bio
            C_Result[("📋 CIVIC Analysis Results<br/>• Gene-level Clinical Interpretations<br/>• Therapeutic Relevance<br/>• Evidence Synthesis")]
            C_Eval --> C_Result
        end
        
        %% PharmGKB System  
        subgraph PharmGKB["💊 PharmGKB Analysis System"]
            direction TB
            P_Bio["BioExpert Agent<br/>(Pharmacogenomic Analysis)"]
            P_Eval["Content Validator<br/>(Quality Check)"]
            P_Bio --> P_Eval
            P_Eval -->|"APPROVED/NOT APPROVED"| P_Bio
            P_Result[("📋 PharmGKB Analysis Results<br/>• Pharmacogenomic Associations<br/>• Drug Response Patterns<br/>• Genetic Variant Effects")]
            P_Eval --> P_Result
        end
        
        %% GProfiler System
        subgraph GProfiler["🔬 Gene Enrichment Analysis System"]
            direction TB
            G_Bio["Gene Enrichment Expert<br/>(Pathway & Function Analysis)"]
            G_Eval["Content Validator<br/>(Quality Check)"]
            G_Bio --> G_Eval
            G_Eval -->|"APPROVED/NOT APPROVED"| G_Bio
            G_Result[("📋 Gene Enrichment Results<br/>• Pathway Enrichment Analysis<br/>• Biological Process Insights<br/>• Functional Annotations")]
            G_Eval --> G_Result
        end
    end
    
    %% Integration Phase
    subgraph Integration["🔄 Evidence Integration Phase"]
        direction TB
        
        subgraph Novelty["🎯 Evidence Integration System"]
            direction TB
            
            subgraph NoveltyAgents["Evidence Integration Agents"]
                direction LR
                Orchestrator["🎯 Orchestrator<br/>Evidence Coordination"]
                ReportComposer["📝 Report Composer<br/>Unified Report Creation"]
                ContentValidator["✅ Content Validator<br/>Structural Integrity"]
                CriticalReviewer["🔍 Critical Reviewer<br/>Bias Analysis"]
                RelevanceValidator["🎯 Relevance Validator<br/>Question Alignment"]
            end
            
            %% Report Composer
            ReportComposer["📝 Report Composer Agent<br/>(Unified Report Creation)"]
            
            %% Feedback Agents (Parallel)
            subgraph Feedback["👥 Feedback Agents (Parallel Review)"]
                direction LR
                ContentValidator["✅ Content Validator<br/>(Structure & Quality)"]
                CriticalReviewer["🔍 Critical Reviewer<br/>(Bias & Analysis)"]
                RelevanceValidator["🎯 Relevance Validator<br/>(Question Alignment)"]
            end
            
            %% Iteration Loop
            Orchestrator --> ReportComposer
            ReportComposer --> Feedback
            Feedback -->|"All APPROVED"| FinalReport[("📊 Final Unified Report<br/>• Novel Biomarkers<br/>• Clinical Implications<br/>• Well-Known Interactions<br/>• Conclusions")]
            Feedback -->|"NOT APPROVED<br/>(Feedback)"| ReportComposer
        end
    end
    
    %% Flow Connections
    Input --> Evidence
    
    %% Parallel execution to Novelty System
    C_Result --> Orchestrator
    P_Result --> Orchestrator  
    G_Result --> Orchestrator
    
    %% Styling
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef civicStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef pharmStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef enrichStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef noveltyStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef resultStyle fill:#f1f8e9,stroke:#33691e,stroke-width:3px
    classDef agentStyle fill:#e3f2fd,stroke:#0d47a1,stroke-width:1px
    
    class Input inputStyle
    class CIVIC,C_Bio,C_Eval,C_Result civicStyle
    class PharmGKB,P_Bio,P_Eval,P_Result pharmStyle
    class GProfiler,G_Bio,G_Eval,G_Result enrichStyle
    class Novelty,Orchestrator,ReportComposer,ContentValidator,CriticalReviewer,RelevanceValidator noveltyStyle
    class FinalReport resultStyle
```

## System Overview

### 🔍 **Phase 1: Parallel Evidence Analysis**
Three independent agent systems analyze pre-extracted evidence simultaneously:

1. **CIVIC System** 🧬
   - Analyzes pre-extracted clinical evidence for genetic variants
   - Produces gene-level clinical interpretations and therapeutic relevance assessments
   - Uses BioExpert + Content Validator agents

2. **PharmGKB System** 💊
   - Processes pre-extracted pharmacogenomic evidence
   - Generates pharmacogenomic associations and drug response pattern analysis
   - Uses BioExpert + Content Validator agents

3. **GProfiler System** 🔬
   - Analyzes gene sets for pathway and functional enrichment
   - Produces pathway enrichment analysis and biological process insights
   - Uses Gene Enrichment Expert + Content Validator agents

### 🔄 **Phase 2: Evidence Integration**
The **Evidence Integration System** 🎯 integrates all evidence:

1. **Orchestrator Agent** 🎭
   - Consolidates evidence from all three systems
   - Coordinates the integration workflow

2. **Report Composer Agent** 📝
   - Creates unified reports with 4 sections:
     - Potential Novel Biomarkers
     - Clinical Implications  
     - Well-Known Interactions
     - Conclusions

3. **Feedback Agents** 👥 (Parallel Review)
   - **Content Validator**: Checks structure and quality
   - **Critical Reviewer**: Identifies biases and alternative interpretations
   - **Relevance Validator**: Ensures question alignment and novelty assessment

### 🔄 **Iterative Refinement**
- If any feedback agent responds "NOT APPROVED", the Report Composer revises the report
- Process continues until all three feedback agents approve
- Maximum iterations prevent infinite loops

### 📊 **Final Output**
Comprehensive unified report addressing the user's research question with evidence from all three biomedical knowledge sources. 