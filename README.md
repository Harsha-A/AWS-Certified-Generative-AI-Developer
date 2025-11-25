# AWS-ML-Services
ML Services


# Comprehensive AWS Generative AI Exam Study Notes

This detailed guide covers all the AWS services tested on the generative AI certification exam, combining foundational concepts with exam-focused technical details.[1][2][3]

## Amazon Bedrock Core Services

### Amazon Bedrock
Fully managed service for building and scaling generative AI applications using foundation models (FMs) without infrastructure management.[3]

**Key Concepts:**
- Provides access to multiple pre-trained foundation models (GPT-based, BERT-based, Claude, Llama) for text generation, summarization, image creation, and code generation
- Supports fine-tuning of FMs with domain-specific data to adapt models for healthcare, finance, customer service without ML expertise
- Pay-as-you-go pricing based on token consumption; Provisioned Throughput option for guaranteed high-availability performance
- Integrates with S3, SageMaker, OpenSearch for end-to-end AI workflows and vector search capabilities
- Supports VPC integration for deploying models in isolated networks meeting compliance standards

**Security & Governance:**
- IAM integration for access control to models and data
- Encryption for data at rest and in transit
- CloudTrail and CloudWatch integration for auditing and monitoring model usage

### Amazon Bedrock AgentCore
Platform for building agentic AI systems that interact with external tools, maintain persistent memory, and execute secure actions.[4]

**Key Concepts:**
- Automates multi-step complex workflows by chaining tasks or models without manual intervention
- Manages data retrieval followed by generative output in customer service workflows
- Secure sandboxed code execution environment for agent actions
- Supports custom configurations for orchestrating tasks across different foundation models and external systems
- Error handling and monitoring capabilities for agent behavior tracking

### Amazon Bedrock Knowledge Bases
Enables retrieval-augmented generation (RAG) by connecting models with enterprise knowledge sources.[3]

**Key Concepts:**
- Supports document chunking strategies for optimal retrieval performance
- Uses Amazon Titan embeddings for vector representation of documents
- Integrates with vector databases (OpenSearch Serverless, Aurora PostgreSQL, Pinecone)
- Hybrid search combining keyword and semantic search capabilities
- Reduces hallucination by grounding model responses in real-world enterprise data

### Amazon Bedrock Prompt Management
Centralized service for designing, storing, versioning, and managing prompt templates.[3]

**Key Concepts:**
- Supports reusable prompt templates with variables for dynamic content
- Version control for prompt evolution and A/B testing
- Integration with Bedrock models for consistent prompt deployment
- Enables prompt optimization through iterative refinement and testing

### Amazon Bedrock Prompt Flows
Visual workflow orchestration for complex multi-step generative AI applications.

**Key Concepts:**
- Sequential prompt execution with conditional branching logic
- Chaining multiple model calls with intermediate processing steps
- Error handling and retry mechanisms for robust workflows
- Integration with Lambda functions for custom processing logic

### Prompt Engineering Techniques
Critical skill for maximizing model accuracy and relevance.[3]

**Techniques:**
- **Zero-shot**: No examples provided, model relies on pre-training
- **Few-shot**: Provide 2-5 examples to guide model behavior
- **Chain-of-thought**: Include reasoning steps in prompts for complex tasks
- Context optimization: Include relevant background information for domain-specific queries

## Amazon SageMaker Ecosystem

### Amazon SageMaker AI
Comprehensive, fully managed service for the entire ML lifecycle from data prep to model monitoring.[3]

**Core Capabilities:**
- Supports TensorFlow, PyTorch, MXNet, and other open-source frameworks
- Handles supervised, unsupervised, and reinforcement learning
- Provides managed infrastructure for training and deployment
- Integrates with EC2 spot instances for cost-optimized training

### SageMaker Unified Studio
Web-based IDE providing unified interface for all ML development tasks.[3]

**Features:**
- Single environment for data prep, training, debugging, and deployment
- Experiment tracking with artifact versioning and model comparison
- Collaboration tools for team-based ML development
- Notebook-based development with GPU/CPU instance selection

### SageMaker Data Wrangler
Simplifies data preparation with visual interface and 300+ built-in transformations.[3]

**Key Features:**
- Missing data handling (imputation, deletion strategies)
- Feature engineering (encoding, scaling, normalization)
- Data visualization for exploratory analysis
- Integration with S3, Redshift, RDS, Athena data sources
- Export transformed data to SageMaker Pipelines or Feature Store

### SageMaker Ground Truth
Creates high-quality training datasets through human and automatic labeling.[3]

**Capabilities:**
- Supports 2D/3D object detection, bounding boxes, semantic segmentation, text classification
- Active learning reduces labeling costs by auto-labeling simple tasks
- Integrated workforce options: Mechanical Turk, private workforce, vendor-managed teams
- Quality control through consensus-based validation and gold standard examples

### SageMaker Clarify
Detects bias in data/models and provides explainability for predictions.[3]

**Key Functions:**
- Pre-training bias detection across sensitive attributes (race, gender, age)
- Post-training bias metrics (disparate impact, demographic parity)
- SHAP (SHapley Additive exPlanations) for global and local model explainability
- Feature importance analysis for understanding model decisions
- Critical for regulatory compliance and responsible AI practices

### SageMaker Model Monitor
Automatically monitors deployed models for data drift and performance degradation.[3]

**Monitoring Capabilities:**
- Tracks accuracy, precision, recall, F1 score over time
- Detects input data distribution changes (data drift)
- Identifies prediction quality degradation
- Configurable CloudWatch alerts for anomaly detection
- Enables automated retraining triggers when thresholds are breached

### SageMaker JumpStart
Pre-built models and solutions for rapid ML project initiation.[3]

**Features:**
- Popular model architectures for NLP, computer vision, time series
- Pre-built solutions: fraud detection, demand forecasting, personalized recommendations, churn prediction
- One-click deployment of foundation models
- Fine-tuning capabilities for domain adaptation

### SageMaker Model Registry
Centralized repository for versioning, cataloging, and deploying ML models.[3]

**Capabilities:**
- Model versioning with metadata tracking (training data, parameters, metrics)
- Approval workflows for model promotion (dev → staging → production)
- Integration with CI/CD pipelines for automated deployment
- Model lineage tracking for audit and compliance

### SageMaker Neo
Optimizes ML models for deployment on edge devices and cloud instances.[3]

**Optimization Features:**
- Compiles models once, runs on multiple hardware platforms (ARM, Intel, NVIDIA)
- Reduces model size and improves inference performance (up to 2x speedup)
- Supports TensorFlow, PyTorch, MXNet, ONNX model formats
- Enables deployment on IoT devices, mobile, edge locations

### SageMaker Processing
Runs data processing and model evaluation workloads at scale.[3]

**Use Cases:**
- Feature engineering on large datasets
- Model evaluation with custom metrics
- Data validation before training
- Batch preprocessing for inference pipelines
- Supports scikit-learn, pandas, custom Docker containers

## Document AI and NLP Services

### Amazon Comprehend
Natural language processing service for text analysis.[3]

**Capabilities:**
- Entity extraction (people, places, organizations, dates)
- Sentiment analysis (positive, negative, neutral, mixed)
- Key phrase extraction for document summarization
- Language detection (100+ languages)
- Topic modeling for document categorization
- Custom entity recognition for domain-specific entities
- PII (Personally Identifiable Information) detection and redaction

### Amazon Textract
Automated document text and data extraction service.[3]

**Features:**
- OCR (Optical Character Recognition) for scanned documents
- Table extraction with cell-level accuracy
- Form parsing (key-value pair extraction)
- Signature detection in forms
- Query-based extraction for specific document sections
- Integration with A2I for human validation of low-confidence results

### Amazon Transcribe
Automatic speech recognition service converting audio to text.

**Key Features:**
- Real-time and batch transcription
- Custom vocabulary for domain-specific terminology
- Speaker identification (speaker diarization)
- Multi-language support with automatic language detection
- Profanity filtering and content redaction
- Medical transcription specialization for clinical documentation

## Computer Vision Services

### Amazon Rekognition
Image and video analysis using deep learning.[3]

**Capabilities:**
- Object and scene detection in images/videos
- Facial analysis (emotion, age range, gender)
- Face comparison and face search
- Celebrity recognition
- Text detection in images (OCR)
- Content moderation (unsafe content detection)
- Custom label training for domain-specific objects
- Integration with A2I for human review of uncertain predictions

## Enterprise Search and Knowledge Management

### Amazon Kendra
Intelligent enterprise search powered by ML for RAG applications.[2][5]

**Key Features:**
- Natural language query understanding with semantic search
- GenAI index optimized for retrieval augmented generation (RAG)
- Integrates with Amazon Q Business and Bedrock Knowledge Bases
- Connectors for 50+ data sources (SharePoint, S3, RDS, Salesforce, ServiceNow)
- Document ranking based on relevance and user permissions
- Faceted search with metadata filtering
- Learning from user interactions to improve results
- Access control list (ACL) support for secure document retrieval

## Conversational AI

### Amazon Lex
Low-code service for building conversational interfaces (chatbots, voice bots).[3]

**Features:**
- Natural language understanding (NLU) and automatic speech recognition (ASR)
- Multi-turn conversation management with context tracking
- Intent and slot recognition for extracting user requirements
- Integration with Lambda for business logic execution
- Multi-language support
- Sentiment analysis during conversations
- Voice and text channel support (phone, web, mobile, messaging platforms)

## Enterprise AI Assistants

### Amazon Q Business
AI-powered enterprise assistant for business workflows and knowledge management.

**Capabilities:**
- Natural language Q&A over enterprise data sources
- RAG-based responses grounded in company documents
- Integration with 40+ enterprise connectors
- User permission inheritance for secure information access
- Conversation history and context management
- Custom plugin development for business logic integration

### Amazon Q Business Apps
Low-code platform for building custom AI-powered business applications.[6]

**Features:**
- Drag-and-drop app builder without coding
- Generative AI capabilities for content creation
- Integration with Q Business for data access
- Custom workflow automation
- Shareable across organization with role-based access

### Amazon Q Developer
AI assistant for software development and AWS resource management.[7]

**Development Features:**
- Inline code completion and generation
- Security scanning for vulnerabilities
- Code explanation and documentation generation
- Debugging assistance and performance optimization
- Query AWS resources, architecture patterns, and documentation
- CLI integration for command generation

## Foundation Models

### Amazon Titan
Amazon's proprietary foundation models for text and embeddings.[3]

**Model Types:**
- **Titan Text**: Large language models for generation, summarization, Q&A
- **Titan Embeddings**: Convert text to vector embeddings for semantic search
- **Titan Image Generator**: Text-to-image generation and image editing
- Optimized for RAG workflows with high-quality embeddings
- Cost-effective compared to third-party models
- Built-in responsible AI guardrails

## Human-in-the-Loop AI

### Amazon Augmented AI (A2I)
Service for integrating human review workflows into ML predictions.[1]

**Key Concepts:**
- Human Loop workflow: Define conditions triggering human review (e.g., confidence < 80%)
- Direct integration with Textract, Rekognition, and SageMaker
- Workflow structure: HumanLoop → Review → Output Consolidation
- Private workforce or Mechanical Turk options

**Common Use Cases:**
- Content moderation requiring human judgment
- Low-confidence prediction validation
- PII redaction verification for compliance
- OCR output correction for medical/clinical documents
- Document classification review in large datasets
- Model drift monitoring through human feedback

## Exam Preparation Best Practices

**Focus Areas:**
- Master RAG architecture patterns with Bedrock Knowledge Bases and Kendra integration
- Understand prompt engineering techniques (zero-shot, few-shot, chain-of-thought)
- Learn SageMaker MLOps tools (Pipelines, Model Monitor, Clarify, Model Registry)
- Practice security configurations (VPC, IAM, encryption) for generative AI deployments
- Study cost optimization strategies (Provisioned Throughput vs on-demand, token management)
- Understand bias detection and model explainability with Clarify and A2I
- Know integration patterns between services (Bedrock + Kendra + Q Business)
- Review multi-modal model capabilities and use cases

**Evaluation Metrics:**
- ROUGE (text summarization quality)
- BLEU (translation accuracy)
- BERTScore (semantic similarity)
- Human evaluation criteria for generative outputs

This comprehensive guide covers the technical depth required for AWS generative AI certification success.[5][2][7][1][3]

[1](https://tutorialsdojo.com/amazon-augmented-ai-a2i/)
[2](https://docs.aws.amazon.com/kendra/latest/dg/what-is-kendra.html)
[3](https://aws.amazon.com/certification/certified-ai-practitioner/)
[4](https://aws.amazon.com/bedrock/agentcore/)
[5](https://www.xenonstack.com/blog/ai-agents-with-amazon-kendra)
[6](https://aws.amazon.com/blogs/training-and-certification/category/amazon-q/)
[7](https://www.pluralsight.com/paths/amazon-q-for-developer)
[8](https://docs.aws.amazon.com/augmented-ai/)
[9](https://docs.aws.amazon.com/sagemaker/latest/dg/a2i-getting-started.html)
[10](https://notes.kodekloud.com/docs/AWS-Solutions-Architect-Associate-Certification/Services-Data-and-ML/Augmented-AI)
[11](https://dev.to/aws/have-you-heard-about-amazon-augmented-ai-434n)
[12](https://sagemaker-examples.readthedocs.io/en/latest/aws_marketplace/using_model_packages/amazon_augmented_ai_with_aws_marketplace_ml_models/amazon_augmented_ai_with_aws_marketplace_ml_models.html)
