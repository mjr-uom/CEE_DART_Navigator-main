# Agent Systems Orchestration Diagram

This diagram shows how the four agent systems work together in the CEE DART Navigator.

```mermaid
flowchart TD
    %% Input Data
    Input[("📊 Input Data<br/>• Context<br/>• Question<br/>• Gene Lists")]
    
    %% Evidence Collection Phase
    subgraph Evidence["🔍 Evidence Collection Phase (Parallel Execution)"]
        direction TB
        
        %% CIVIC System
        subgraph CIVIC["🧬 CIVIC Analysis System"]
            direction TB
            C_Bio["BioExpert Agent<br/>(Clinical Evidence)"]
            C_Eval["Content Validator<br/>(Quality Check)"]
            C_Bio --> C_Eval
            C_Eval -->|"APPROVED/NOT APPROVED"| C_Bio
            C_Result[("📋 CIVIC Results<br/>• Clinical Variants<br/>• Drug Associations<br/>• Evidence Levels")]
            C_Eval --> C_Result
        end
        
        %% PharmGKB System  
        subgraph PharmGKB["💊 PharmGKB Analysis System"]
            direction TB
            P_Bio["BioExpert Agent<br/>(Pharmacogenomic Evidence)"]
            P_Eval["Content Validator<br/>(Quality Check)"]
            P_Bio --> P_Eval
            P_Eval -->|"APPROVED/NOT APPROVED"| P_Bio
            P_Result[("📋 PharmGKB Results<br/>• Drug-Gene Interactions<br/>• Pharmacokinetics<br/>• Dosing Guidelines")]
            P_Eval --> P_Result
        end
        
        %% GProfiler System
        subgraph GProfiler["🔬 Gene Enrichment Analysis System"]
            direction TB
            G_Bio["Gene Enrichment Expert<br/>(Pathway Analysis)"]
            G_Eval["Content Validator<br/>(Quality Check)"]
            G_Bio --> G_Eval
            G_Eval -->|"APPROVED/NOT APPROVED"| G_Bio
            G_Result[("📋 Enrichment Results<br/>• Biological Pathways<br/>• GO Terms<br/>• Functional Networks")]
            G_Eval --> G_Result
        end
    end
    
    %% Integration Phase
    subgraph Integration["🔄 Evidence Integration Phase"]
        direction TB
        
        subgraph Novelty["🎯 Novelty Analysis System"]
            direction TB
            
            %% Orchestrator
            Orchestrator["🎭 Orchestrator Agent<br/>(Evidence Consolidation)"]
            
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

### 🔍 **Phase 1: Parallel Evidence Collection**
Three independent agent systems run simultaneously:

1. **CIVIC System** 🧬
   - Analyzes clinical evidence for genetic variants
   - Focuses on drug associations and therapeutic relevance
   - Uses BioExpert + Content Validator agents

2. **PharmGKB System** 💊
   - Processes pharmacogenomic evidence
   - Examines drug-gene interactions and dosing guidelines
   - Uses BioExpert + Content Validator agents

3. **GProfiler System** 🔬
   - Performs gene enrichment and pathway analysis
   - Identifies biological processes and functional networks
   - Uses Gene Enrichment Expert + Content Validator agents

### 🔄 **Phase 2: Evidence Integration**
The **Novelty Analysis System** 🎯 integrates all evidence:

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