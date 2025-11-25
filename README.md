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



----------



# AWS Management and Governance Services for Generative AI Exam Study Notes

This guide provides detailed exam-focused coverage of AWS Management and Governance services critical for monitoring, optimizing, and managing generative AI workloads.[1][2][3][4]

## Monitoring and Observability

### Amazon CloudWatch
Comprehensive monitoring and observability service for AWS resources, applications, and ML models.[2][1]

**Core Capabilities:**
- Metrics repository collecting and storing performance data from AWS services and custom applications
- Default metrics for most AWS services (CPU, network, disk, status checks) without additional configuration
- Custom metrics for application-specific monitoring (model accuracy, token usage, inference latency)
- Metric resolution: Standard (5 minutes) or High-resolution (1 second)
- Metric retention: 15 months for standard resolution

**Key Features for AI/ML Workloads:**
- **SageMaker Integration**: Automatically collects ModelLatency, Invocations, InvocationsPerInstance, 4XXErrors, 5XXErrors metrics[5][2]
- **Bedrock Monitoring**: Tracks token consumption, API latency, throttling events, model invocation counts
- **Real-time Dashboards**: Visualize training metrics (loss, accuracy) in near real-time for ML experiments[5]
- **Percentile Statistics**: Track p50, p90, p99 latency for inference endpoints to understand tail performance
- **Anomaly Detection**: ML-powered anomaly detection for automatic baseline creation and alerting

**Alarms and Actions:**
- Create alarms on metric thresholds (e.g., ModelLatency > 500ms)
- Composite alarms combining multiple conditions (high latency AND high error rate)
- Alarm actions: SNS notifications, Auto Scaling policies, Lambda functions, Systems Manager automation

**CloudWatch for Cost Monitoring:**
- Track compute utilization (EC2, SageMaker instances) to identify underutilized resources
- Monitor Bedrock token consumption to optimize prompt engineering
- Set billing alarms for unexpected spending patterns

### Amazon CloudWatch Logs
Centralized log management for application, system, and service logs.[1][2]

**Key Concepts:**
- **Log Groups**: Container for log streams (e.g., `/aws/sagemaker/Endpoints/my-endpoint`)
- **Log Streams**: Sequence of log events from same source (e.g., individual container instances)
- **Log Events**: Individual log entries with timestamp and message
- Retention policies: 1 day to 10 years, or indefinite

**AI/ML Log Sources:**
- SageMaker training job logs (stdout/stderr from training containers)
- SageMaker endpoint invocation logs (request/response payloads)
- Bedrock API call logs (prompts, completions, metadata)
- Lambda function logs for custom processing logic
- Application logs from containerized AI services

**CloudWatch Logs Insights:**
- Interactive query language for analyzing log data
- Common queries: Error analysis, latency patterns, user behavior tracking
- Visualizations: Time series charts, bar graphs, tables
- Sample query for SageMaker errors: `fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc`

**Integration with AI Workflows:**
- Install CloudWatch Agent on EC2/ECS for custom application logs
- Enable SageMaker endpoint data capture for logging inference requests/responses
- Export logs to S3 for long-term archival and compliance
- Stream logs to Lambda for real-time processing and alerting

### Amazon CloudWatch Synthetics
Creates canaries that monitor endpoints and APIs with automated tests.[2]

**Key Features:**
- Scheduled synthetic transactions simulating user behavior
- HTTP/HTTPS endpoint monitoring with custom scripts (Node.js, Python)
- Visual monitoring: Screenshot capture for UI regression detection
- Canary types: Heartbeat (availability), API canary (response validation), Broken link checker, GUI workflow

**AI/ML Use Cases:**
- Monitor SageMaker real-time endpoint availability 24/7
- Validate Bedrock API response quality and latency
- Test end-to-end AI application workflows (user query → RAG → response)
- Alert on failures with CloudWatch Alarms integrated with SNS/Lambda

**Best Practices:**
- Create canaries calling inference endpoints with sample requests
- Set alarm thresholds for canary failure rate (e.g., >10% failures in 5 minutes)
- Use canary metrics to correlate availability issues with deployment changes

### AWS CloudTrail
Governance, compliance, and audit service tracking API activity across AWS account.[2]

**Core Functionality:**
- Records all API calls made to AWS services (console, CLI, SDK, automation tools)
- Captures: Identity (who), timestamp (when), source IP (where), action (what), resources (target)
- Event types: Management events (control plane), Data events (data plane like S3 object access)

**AI/ML Audit and Security:**
- Track who accessed or modified SageMaker models, Bedrock configurations, training data
- Monitor IAM role assumptions for ML workload security analysis
- Detect unauthorized API calls to sensitive AI services
- Compliance: Maintain immutable audit trail for regulatory requirements (HIPAA, GDPR, SOC 2)

**Integration:**
- Deliver logs to S3 for long-term storage and analysis
- Stream events to CloudWatch Logs for real-time monitoring
- Integrate with EventBridge for automated responses to specific API activities

**CloudTrail vs CloudWatch:**
- **CloudTrail**: Who did what, when? (API audit trail, governance)
- **CloudWatch**: How is the system performing? (metrics, logs, operational monitoring)

### Amazon Managed Grafana
Fully managed service for Grafana, providing data visualization and analytics dashboards.

**Key Features:**
- Pre-built dashboards for AWS services (CloudWatch, X-Ray, Prometheus)
- Multi-account and multi-region monitoring in unified interface
- Plugin ecosystem for extending visualization capabilities
- User authentication via AWS SSO, SAML, OAuth

**AI/ML Monitoring:**
- Visualize SageMaker training metrics across multiple experiments
- Compare model performance over time with custom queries
- Monitor distributed training cluster resource utilization
- Create operational dashboards combining CloudWatch metrics, logs, and traces

## Auto Scaling and Automation

### AWS Auto Scaling
Unified scaling for multiple AWS services to optimize performance and cost.

**Supported Services for AI/ML:**
- SageMaker endpoint instances (inference auto-scaling)
- EC2 instances for distributed training clusters
- ECS/Fargate tasks for containerized AI applications
- DynamoDB tables for vector store scaling

**SageMaker Endpoint Auto Scaling:**
- Target tracking: Scale based on InvocationsPerInstance metric
- Configure min/max instance counts and target metric value
- Scale-out cooldown: Wait time before adding more instances (default 300s)
- Scale-in cooldown: Wait time before removing instances (default 300s)
- Protects against rapid scaling fluctuations and cost spikes

**Best Practices:**
- Set conservative scale-in cooldown for ML endpoints (5-10 minutes) to avoid cold starts
- Monitor scaling activities in CloudWatch to optimize thresholds
- Use scheduled scaling for predictable traffic patterns (e.g., business hours)
- Combine with Provisioned Throughput for Bedrock to guarantee capacity during high demand

### AWS Systems Manager
Unified interface for operational data and automation across AWS resources.

**Key Components:**
- **Parameter Store**: Secure, hierarchical storage for configuration data and secrets (API keys, model URIs)
- **Session Manager**: Secure shell access to EC2/on-premises instances without SSH keys
- **Patch Manager**: Automated OS patching for EC2 fleets
- **Run Command**: Execute commands on multiple instances simultaneously
- **State Manager**: Maintain consistent instance configurations

**AI/ML Use Cases:**
- Store and version ML model hyperparameters in Parameter Store
- Automate SageMaker training job submission via Run Command
- Maintain consistent software dependencies on training instance fleets
- Secure access to notebook instances without exposing SSH ports

## Cost Management

### AWS Cost Explorer
Visualize, understand, and manage AWS costs and usage over time.[3]

**Core Features:**
- Interactive charts showing cost trends by service, region, usage type
- Filtering and grouping by multiple dimensions (service, tag, instance type)
- Forecasting: Predict future costs based on historical patterns
- Cost allocation tags for tracking AI project spending
- Rightsizing recommendations for underutilized resources

**AI/ML Cost Analysis:**
- Identify expensive services (SageMaker training, Bedrock token usage, S3 storage)
- Compare training costs across different instance types (ml.p4d vs ml.p3)
- Track Bedrock API costs by model (Claude vs Titan) to optimize model selection
- Analyze cost trends after implementing optimization strategies

**Reservations and Savings Plans:**
- SageMaker Savings Plans: Up to 64% discount for committed usage
- EC2 Savings Plans: Cover training/inference instance costs
- Reserved Capacity: Guarantee SageMaker notebook/endpoint instance availability

### AWS Cost Anomaly Detection
ML-powered service identifying unusual spending patterns and root causes.[6][3]

**How It Works:**
- Uses machine learning to establish spending baselines across services
- Continuously monitors actual spend against predicted patterns
- Detects anomalies using rolling 24-hour windows for faster identification[3]
- Sends alerts via email, SNS, or Slack when anomalies detected

**Enhanced Detection Algorithm (Nov 2025):**
- Compares current costs against equivalent 24-hour periods from previous days
- Removes delay from incomplete calendar-day comparisons
- Accounts for workloads with different morning/evening usage patterns[3]
- Reduces false positives by contextual time-of-day analysis

**Configuration:**
- Define cost monitors: Entire account, specific services, or cost allocation tags
- Set alert threshold: Dollar amount ($100) or percentage (20% increase)
- Choose notification channels: Email, SNS topics for Slack/PagerDuty integration
- Segment by: Service, linked account, cost category, tag

**AI/ML Cost Anomaly Scenarios:**
- Unexpected SageMaker training job costs from instance type misconfiguration
- Bedrock token usage spikes from inefficient prompts or RAG loops
- S3 storage growth from unmanaged training data or model artifacts
- Inference endpoint running 24/7 instead of on-demand schedule

**Limitations:**
- Requires 10-14 days of usage data to establish baseline
- Manual configuration of monitors and segments
- No unit cost analysis (per-customer, per-project granularity)[6]
- Best for gross anomalies, not fine-grained cost optimization

## Communication and Collaboration

### AWS Chatbot
Interactive agent enabling ChatOps for AWS services via Slack, Microsoft Teams, Amazon Chime.

**Key Features:**
- Receive CloudWatch alarms, AWS Health notifications, Security Hub findings in chat
- Execute AWS CLI commands directly from chat (read-only or admin actions)
- Configure notification routing by severity, service, or tag
- IAM role-based permissions control chat command execution

**AI/ML ChatOps:**
- Receive alerts when SageMaker training jobs fail or complete
- Get notified of Bedrock API throttling or quota limits
- Query CloudWatch metrics from chat: `@aws cloudwatch get-metric-statistics`
- Acknowledge and resolve incidents collaboratively in team channels

**Security Considerations:**
- Use least-privilege IAM roles for Chatbot
- Enable audit logging of commands executed via chat
- Restrict admin actions to specific channels or users

## Service Management

### AWS Service Catalog
Centralized governance for IT services, enabling standardized provisioning of approved resources.

**Key Concepts:**
- **Products**: CloudFormation templates for AWS resources (e.g., SageMaker notebook with approved configuration)
- **Portfolios**: Collections of products with access controls
- **Constraints**: Rules limiting product configuration (instance types, regions)
- **Provisioned Products**: Launched instances of catalog products

**AI/ML Governance:**
- Create approved SageMaker notebook configurations with pre-installed libraries
- Standardize training job templates with security controls (VPC, encryption)
- Enforce cost guardrails by limiting instance types (ml.m5 family only)
- Provide self-service access to ML infrastructure without granting direct IAM permissions

**Benefits:**
- Consistent infrastructure deployment across teams
- Centralized version control for ML environment templates
- Compliance enforcement through constraints and launch rules
- Audit trail of provisioned resources for governance

## AWS Well-Architected Tool

### Core Framework
Provides architectural best practices across six pillars for evaluating workloads.[7][4]

**Six Pillars:**
1. **Operational Excellence**: Monitor, operate, and continuously improve processes
2. **Security**: Protect information, systems, and assets
3. **Reliability**: Recover from failures, scale to meet demand
4. **Performance Efficiency**: Use resources efficiently to meet requirements
5. **Cost Optimization**: Achieve business outcomes at lowest price point
6. **Sustainability**: Minimize environmental impact of cloud workloads

### Generative AI Lens
Specialized lens extending Well-Architected Framework for generative AI applications.[4]

**Operational Excellence for GenAI:**
- Achieve consistent model output quality through evaluation frameworks (ROUGE, BLEU, human feedback)
- Monitor operational health: Token usage, latency, error rates, model drift
- Maintain traceability: Log prompts, responses, and model versions for debugging
- Automate lifecycle management: CI/CD pipelines for model deployment and updates
- Determine when to execute model customization: Fine-tuning vs prompt engineering decisions

**Security for GenAI:**
- Protect endpoints: VPC isolation, encryption in transit/at rest, IAM least privilege
- Mitigate harmful outputs: Content filtering, guardrails, human review workflows (A2I)
- Monitor and audit events: CloudTrail API logging, CloudWatch metrics for anomalous behavior
- Secure prompts: Prevent prompt injection attacks, validate user inputs
- Remediate model poisoning risks: Data validation, provenance tracking, regular retraining

**Reliability for GenAI:**
- Handle throughput requirements: Auto-scaling, Provisioned Throughput for Bedrock
- Maintain reliable component communication: Retry logic, circuit breakers, graceful degradation
- Implement observability: Distributed tracing with X-Ray, structured logging
- Handle failures gracefully: Fallback models, cached responses, error messaging
- Version artifacts: Model registry, prompt versioning, dataset lineage

**Performance Efficiency for GenAI:**
- Capture and improve model performance: A/B testing, continuous evaluation
- Maintain acceptable performance: Latency budgets, caching strategies, batch processing
- Optimize computation resources: Instance type selection, model quantization, SageMaker Neo
- Improve data retrieval: Vector store optimization, hybrid search, chunking strategies for RAG

**Cost Optimization for GenAI:**
- Select cost-optimized models: Compare cost per token across Bedrock models
- Balance cost and performance of inference: Provisioned vs on-demand, batch vs real-time
- Engineer prompts for cost: Minimize token count, use caching, avoid redundant context
- Optimize vector stores: Right-size databases, implement TTL policies for embeddings
- Optimize agent workflows: Reduce tool invocations, implement result caching

**Sustainability for GenAI:**
- Minimize computational resources for training: Transfer learning, efficient architectures
- Optimize customization: Use parameter-efficient fine-tuning (PEFT) instead of full fine-tuning
- Reduce hosting footprint: Model quantization, instance rightsizing, auto-scaling policies
- Efficient data processing: Incremental updates, deduplication, compression
- Leverage serverless: Lambda for intermittent workloads, Bedrock for managed inference

### Using the Well-Architected Tool

**Review Process:**
1. Define workload in AWS WA Tool (name, description, environment)
2. Apply relevant lenses (AWS Well-Architected Framework + Generative AI Lens)
3. Answer questions across all pillars with team collaboration
4. Identify high/medium risk issues (HRIs/MRIs) based on best practice gaps
5. Generate improvement plan with prioritized remediation actions
6. Track progress over time with milestone comparisons

**Generative AI Lens Availability:**
- Download from AWS Well-Architected custom lens GitHub repository
- Import as custom lens into AWS WA Tool
- Available for all AWS accounts at no additional charge

## Exam Preparation Focus Areas

**Monitoring Best Practices:**
- Know which CloudWatch metrics are critical for ML endpoint monitoring (ModelLatency, 5XX errors)[2]
- Understand when to use CloudWatch Logs vs CloudTrail vs X-Ray
- Configure CloudWatch Alarms with appropriate thresholds and actions
- Implement CloudWatch Synthetics canaries for endpoint availability testing

**Cost Optimization Strategies:**
- Use Cost Anomaly Detection for proactive spending alerts[3]
- Leverage Cost Explorer for historical analysis and forecasting
- Apply Savings Plans and Reserved Capacity for predictable workloads
- Tag resources consistently for cost allocation and chargeback

**Automation and Governance:**
- Implement Service Catalog for standardized ML environment provisioning
- Use Systems Manager Parameter Store for configuration management
- Configure Auto Scaling for SageMaker endpoints with appropriate cooldown periods
- Apply Well-Architected Tool reviews regularly for continuous improvement[4]

**Security and Compliance:**
- Enable CloudTrail for comprehensive API audit logging
- Implement least-privilege IAM policies for ML workloads
- Use VPC endpoints for private connectivity to AWS services
- Apply encryption at rest and in transit for all AI/ML data

This comprehensive guide covers the management and governance services essential for operating production generative AI workloads on AWS according to best practices.[1][4][2][3]

[1](https://tutorialsdojo.com/amazon-cloudwatch/)
[2](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/AWS-Machine-Learning-Associate-Practice-Exams)
[3](https://aws.amazon.com/about-aws/whats-new/2025/11/aws-cost-anomaly-detection-accelerates-anomaly/)
[4](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/generative-ai-lens.html)
[5](https://aws.amazon.com/blogs/machine-learning/use-amazon-cloudwatch-custom-metrics-for-real-time-monitoring-of-amazon-sagemaker-model-performance/)
[6](https://spot.io/resources/aws-cost-optimization/aws-cost-anomaly-detection-pros-cons-and-how-to-get-started/)
[7](https://docs.aws.amazon.com/wellarchitected/latest/userguide/lenses.html)
[8](https://docs.aws.amazon.com/machine-learning/latest/dg/cw-doc.html)
[9](https://aws.amazon.com/blogs/training-and-certification/category/management-tools/amazon-cloudwatch/)
[10](https://www.geeksforgeeks.org/cloud-computing/introduction-to-amazon-cloudwatch/)


----------

# AWS Migration, Transfer, Networking, and Content Delivery for Generative AI Exam Study Notes

This comprehensive guide covers AWS networking, content delivery, and migration services critical for deploying, scaling, and securing generative AI workloads.[1][2][3][4][5]

## Migration and Transfer Services

### AWS DataSync
Secure, online service that automates and accelerates data transfer between on-premises and AWS storage services.[3][6]

**Core Capabilities:**
- Automates data copy, scheduling, monitoring, and validation without manual scripting
- Transfers up to 10x faster than open-source tools through network optimization
- Supports incremental transfers for ongoing data synchronization
- Built-in data validation ensures integrity during transfer
- Bandwidth throttling to avoid saturating network connections

**Supported Locations:**
- On-premises: NFS, SMB, HDFS, object storage
- AWS: S3, EFS, FSx for Windows File Server, FSx for Lustre, FSx for OpenZFS, FSx for NetApp ONTAP
- Cross-account and cross-region transfers[7]

**AI/ML Use Cases:**
- **Training Data Migration**: Transfer large datasets from on-premises storage to S3 for SageMaker training[8]
- **Data Lake Consolidation**: Aggregate datasets (Common Crawl, SEC filings) for ML model development[8]
- **Cross-Account ML Workflows**: Transfer training data between development and production accounts
- **Continuous Data Ingestion**: Sync streaming data sources to S3 for real-time model training
- **Backup and DR**: Replicate model artifacts, training datasets, and notebooks to secondary regions

**Configuration Best Practices:**
- Deploy DataSync agent on-premises or in EC2 for network file system access
- Use VPC endpoints for private connectivity without internet gateway
- Configure filters to exclude temporary files or specific directories
- Schedule transfers during off-peak hours to minimize business impact
- Enable CloudWatch logging for monitoring transfer progress and troubleshooting

**Cross-Account Transfers:**
- Create IAM role in source account with S3 read permissions[8]
- Configure destination S3 bucket policy to allow source account IAM role
- Use DataSync Terraform modules for automated, repeatable configurations[8]

### AWS Transfer Family
Fully managed service providing secure file transfers into and out of AWS storage services.

**Supported Protocols:**
- SFTP (SSH File Transfer Protocol)
- FTPS (File Transfer Protocol over SSL)
- FTP (File Transfer Protocol)
- AS2 (Applicability Statement 2)

**Integration with S3/EFS:**
- Direct transfer to S3 buckets or EFS file systems
- Custom identity provider integration (Active Directory, LDAP)
- IAM role-based access control for per-user permissions

**AI/ML Use Cases:**
- Receive training data from external partners via secure SFTP
- Enable data scientists to upload datasets without AWS console access
- Automate data ingestion pipelines with Lambda triggers on file arrival
- Comply with regulatory requirements for secure data exchange

## Networking and Content Delivery

### Amazon API Gateway
Fully managed service for creating, publishing, and managing REST, HTTP, and WebSocket APIs.[9][1]

**API Types:**
- **REST API**: Request/response model with full API lifecycle management
- **HTTP API**: Lower latency, lower cost alternative for simple proxying
- **WebSocket API**: Real-time bidirectional communication for streaming responses

**Integration with AI/ML Services:**
- **SageMaker Endpoint Integration**: Create public REST API fronting inference endpoints[1][9]
- **Direct Integration**: Use mapping templates to invoke SageMaker runtime without Lambda intermediary[9]
- **Lambda Proxy**: Invoke Lambda function that calls SageMaker/Bedrock for additional processing[1]
- **Bedrock API Gateway**: Expose foundation models via REST endpoints with authentication

**Key Features:**
- Request/response transformation with mapping templates (VTL)
- Throttling and rate limiting (burst and steady-state limits)
- API keys for client identification and usage tracking
- Usage plans for tiering access (free tier, paid tier)
- Caching responses to reduce backend load and latency
- Request validation to reject malformed requests before backend invocation

**Security Options:**
- IAM authorization for AWS-signed requests
- Lambda authorizers for custom authentication logic (JWT, OAuth)
- Cognito user pools for user-based access control
- API keys for simple identification (not recommended as sole security)
- Resource policies for VPC endpoint or IP-based restrictions
- Private APIs accessible only from VPC via VPC endpoints

**Monitoring and Logging:**
- CloudWatch metrics: API calls, latency, 4XX/5XX errors, cache hit/miss
- CloudWatch Logs: Request/response logging with configurable detail levels
- X-Ray integration for distributed tracing and performance analysis

**AI/ML Architecture Pattern:**
```
Client → API Gateway → Lambda → SageMaker Endpoint → Response
Client → API Gateway (direct) → SageMaker Runtime API → Response
Client → API Gateway → Lambda → Bedrock API → RAG → Response
```

**Best Practices:**
- Use Lambda authorizers for validating tokens before expensive inference calls
- Enable caching for repeated queries to reduce costs and latency
- Implement throttling to protect endpoints from traffic spikes
- Use stage variables for environment-specific configurations (dev/prod endpoints)

### AWS AppSync
Fully managed GraphQL API service with real-time and offline capabilities.[5][10]

**Core Features:**
- GraphQL API creation with schema-first development
- Real-time subscriptions via WebSockets for live updates
- Offline data synchronization for mobile applications
- Built-in authentication with Cognito, IAM, OIDC, API keys

**AI Gateway Capabilities:**
- **Amazon Bedrock Integration**: Direct data source for synchronous model invocations (≤10 seconds)[10]
- **Asynchronous AI Workflows**: Trigger long-running generative AI tasks with subscription-based progressive updates[10]
- **Multi-Source Data**: Combine AI model responses with database queries (DynamoDB, Aurora) in single GraphQL request[5]
- **Federation**: Merge multiple GraphQL APIs (data sources + AI models) into unified supergraph

**Use Cases for Generative AI:**
- Real-time chatbot interfaces with streaming responses from Bedrock
- Content generation dashboards combining user data and AI outputs
- Multi-tenant AI applications with user-specific model access
- Progressive disclosure of long-running RAG query results

**AppSync Resolvers:**
- VTL (Velocity Template Language) or JavaScript resolvers
- Direct integration with AWS services (Lambda, DynamoDB, Bedrock, HTTP endpoints)
- Pipeline resolvers for multi-step operations (authentication → RAG → model invocation)

### Amazon CloudFront
Global content delivery network (CDN) caching content at edge locations for low latency.[11][12]

**Core Concepts:**
- **Edge Locations**: 450+ Points of Presence (PoPs) worldwide for content caching[11]
- **Regional Edge Caches**: Intermediate cache layer between edge locations and origin
- **Pull-Through Cache**: Content cached on first request, served from cache on subsequent requests[12]
- **TTL (Time to Live)**: Controls how long content stays cached before revalidation

**AI/ML Use Cases:**
- **Model Serving at Edge**: Cache inference responses for popular queries (e.g., product recommendations)
- **Static Asset Delivery**: Serve UI assets for AI applications (React dashboards, chatbot interfaces)
- **API Acceleration**: Cache API Gateway responses for read-heavy AI APIs
- **Lambda@Edge**: Execute custom logic at edge locations for request/response manipulation

**CloudFront Functions vs Lambda@Edge:**
- **CloudFront Functions**: Lightweight JavaScript (<1ms) for header manipulation, URL rewrites, cache key normalization[11]
- **Lambda@Edge**: Full Lambda runtime for complex logic (authentication, A/B testing, content generation)

**Caching Strategies for AI:**
- Cache GET requests to inference endpoints with query parameters as cache keys
- Set appropriate TTL based on model update frequency (e.g., 1 hour for dynamic models)
- Use cache invalidation when models are updated or retrained
- Implement cache headers (Cache-Control, ETag) for conditional requests

**Security Features:**
- Signed URLs/Cookies for restricting content access
- AWS WAF integration for DDoS protection and request filtering
- Field-level encryption for sensitive request data
- HTTPS enforcement with custom SSL certificates

**Best Practices:**
- Use CloudFront with S3 origin for serving trained model artifacts
- Enable origin shield to reduce origin load from multiple edge locations
- Configure custom error pages for graceful degradation when endpoints fail
- Monitor cache hit ratio in CloudWatch to optimize caching effectiveness

### Elastic Load Balancing (ELB)
Distributes incoming traffic across multiple targets for high availability and fault tolerance.[4]

**Load Balancer Types:**
- **Application Load Balancer (ALB)**: HTTP/HTTPS traffic with advanced routing (Layer 7)
- **Network Load Balancer (NLB)**: TCP/UDP traffic with ultra-low latency (Layer 4)
- **Gateway Load Balancer**: Third-party virtual appliances (firewalls, intrusion detection)

**AI/ML Load Balancing:**
- **SageMaker Endpoint Scaling**: Distribute inference requests across multiple endpoint instances[4]
- **Multi-AZ Deployment**: Route traffic across Availability Zones for resilience[4]
- **Container-based Inference**: Load balance ECS/EKS pods running custom ML models
- **Canary Deployments**: Route percentage of traffic to new model versions for A/B testing

**ALB Features for AI:**
- Path-based routing: `/model-v1` → Endpoint A, `/model-v2` → Endpoint B
- Host-based routing: `model-a.example.com` → Endpoint A
- Header-based routing: Route based on API version or client type
- Weighted target groups: 90% to stable model, 10% to experimental model

**Health Checks:**
- Configure health check endpoint (e.g., `/health`) on inference servers
- Set unhealthy threshold (consecutive failures) and healthy threshold (consecutive successes)
- Automatically remove unhealthy targets from rotation
- Integrate with CloudWatch alarms for automated recovery

**Sticky Sessions:**
- Enable session affinity for stateful inference (conversation context)
- Use application-based cookies for user-specific routing
- Balance between stickiness and even load distribution

**Best Practices:**
- Use NLB for latency-sensitive real-time inference (<10ms overhead)
- Use ALB for HTTP-based inference with advanced routing requirements
- Enable cross-zone load balancing for even distribution across AZs[4]
- Monitor UnHealthyHostCount metric to detect endpoint failures

### AWS Global Accelerator
Network layer service improving global application availability and performance.

**How It Works:**
- Provides two static Anycast IP addresses as entry points
- Routes traffic over AWS global network (not public internet)
- Automatically routes to optimal regional endpoint based on health and proximity
- Instant failover to healthy endpoints (30-second detection)

**Benefits for AI/ML:**
- Consistent low-latency access to SageMaker endpoints from global users
- Instant regional failover for mission-critical inference services
- Static IPs simplify firewall whitelisting for enterprise clients
- Performance boost (up to 60%) compared to internet routing

**Use Cases:**
- Multi-region AI application deployment with automatic traffic routing
- Gaming AI (recommendations, matchmaking) requiring <100ms latency
- Financial services AI (fraud detection) with high availability requirements

### AWS PrivateLink
Establishes private connectivity between VPCs and AWS services without internet gateway.[2][13]

**Core Concepts:**
- VPC Interface Endpoints: ENIs in your VPC for accessing AWS services privately
- VPC Gateway Endpoints: Routes in route table for S3 and DynamoDB
- Endpoint Services: Expose your own services to other VPCs via PrivateLink

**AI/ML Security Use Cases:**
- **Bedrock VPC Endpoints**: Access foundation models without internet exposure[13][2]
- **SageMaker VPC Mode**: Train models and run inference entirely within VPC
- **S3 VPC Endpoint**: Access training data in S3 without public internet
- **Secrets Manager Endpoint**: Retrieve API keys for external AI services privately

**Bedrock PrivateLink Integration:**
- Protect model customization jobs using VPC endpoints[2]
- Secure batch inference jobs with private connectivity
- Access Bedrock Knowledge Bases and OpenSearch Serverless via interface endpoints[2]
- Secure ingress to Bedrock AgentCore Gateway through VPC endpoints[13]

**Configuration:**
- Create interface endpoint for desired service (e.g., `com.amazonaws.us-east-1.bedrock-runtime`)
- Associate endpoint with subnets in multiple AZs for high availability
- Configure security groups to allow inbound traffic from application subnets
- Enable private DNS to use standard service endpoints (e.g., `bedrock-runtime.us-east-1.amazonaws.com`)

**Benefits:**
- Data never traverses public internet (compliance requirement)
- Reduced data transfer costs for inter-VPC communication
- Enhanced security posture with network isolation
- Simplified network architecture without NAT gateways

### Amazon Route 53
Scalable DNS web service with health checking and traffic routing capabilities.

**DNS Routing Policies:**
- **Simple**: Single resource (e.g., one ALB for SageMaker endpoints)
- **Weighted**: Percentage-based traffic splitting for A/B testing (80% model-v1, 20% model-v2)
- **Latency**: Route to region with lowest latency for global inference
- **Failover**: Primary/secondary routing for disaster recovery
- **Geolocation**: Route based on user location (EU users → eu-west-1)
- **Geoproximity**: Route based on distance with bias adjustment
- **Multi-value Answer**: Return multiple healthy endpoints with client-side selection

**AI/ML Use Cases:**
- **Multi-Region Inference**: Route users to nearest regional SageMaker endpoint
- **Active-Active Deployment**: Distribute load across multiple regions with latency routing
- **Disaster Recovery**: Failover to secondary region if primary endpoint unhealthy
- **Model Version Management**: Use weighted routing for gradual rollout of new models

**Health Checks:**
- HTTP/HTTPS/TCP health checks with customizable intervals
- String matching for validating response content
- Calculated health checks combining multiple child health checks
- CloudWatch alarm-based health checks for custom metrics

**Private Hosted Zones:**
- DNS resolution for resources within VPC
- Enable internal service discovery for microservices architecture
- Route `model-service.internal` to ECS tasks running inference

### Amazon VPC (Virtual Private Cloud)
Isolated virtual network for launching AWS resources with complete control over networking.

**Core Components:**
- **Subnets**: IP address ranges subdividing VPC (public/private)
- **Route Tables**: Control traffic routing within VPC and to internet
- **Internet Gateway**: Enable public internet access for public subnets
- **NAT Gateway**: Allow private subnets to initiate outbound internet connections
- **Security Groups**: Stateful firewall at instance level
- **Network ACLs**: Stateless firewall at subnet level

**AI/ML VPC Architecture:**
```
Public Subnet: ALB (inference endpoint) → Internet Gateway
Private Subnet: SageMaker Endpoints, ECS Tasks, Lambda Functions
Data Subnet: RDS (metadata), OpenSearch (vector store), S3 VPC Endpoint
```

**VPC Best Practices for AI:**
- Deploy SageMaker training jobs in VPC for accessing private data sources
- Use private subnets for inference endpoints with ALB in public subnet
- Configure VPC endpoints for S3, Bedrock, Secrets Manager to avoid NAT costs
- Implement security groups restricting inference endpoint access to API Gateway or ALB
- Enable VPC Flow Logs for monitoring network traffic patterns and security analysis

**VPC Peering:**
- Connect VPCs across accounts or regions for multi-account ML workflows
- Access centralized ML platform VPC from multiple application VPCs
- Non-transitive: Must create peering connections between each VPC pair

**Transit Gateway:**
- Hub-and-spoke model for connecting multiple VPCs
- Simplifies network architecture for large ML platform deployments
- Centralized routing and monitoring for all VPC traffic

## Exam Preparation Focus Areas

**Networking Patterns:**
- Understand VPC endpoint types and when to use each for AI services[13][2]
- Know how to configure API Gateway with SageMaker endpoints (direct vs Lambda proxy)[9][1]
- Learn CloudFront caching strategies for inference response optimization
- Master load balancing configurations for multi-AZ AI deployments[4]

**Security and Compliance:**
- Configure PrivateLink for private Bedrock and SageMaker access[2]
- Implement API Gateway authentication mechanisms (IAM, Cognito, Lambda authorizers)
- Use security groups and NACLs to restrict inference endpoint access
- Enable CloudTrail and VPC Flow Logs for audit compliance

**Performance Optimization:**
- Use Global Accelerator for global low-latency inference access
- Implement CloudFront caching to reduce inference costs and latency
- Configure Route 53 latency routing for multi-region deployments
- Optimize API Gateway with caching and throttling configurations

**Data Transfer and Migration:**
- Use DataSync for large-scale training data migration to S3[8]
- Configure Transfer Family for secure partner data exchange
- Understand cross-account transfer patterns with IAM roles[8]
- Schedule transfers during off-peak hours to minimize network impact

**GraphQL and Real-Time AI:**
- Leverage AppSync for real-time generative AI applications with subscriptions[10]
- Integrate Bedrock as AppSync data source for synchronous invocations[10]
- Build federated GraphQL APIs combining multiple AI models and data sources[5]

This comprehensive guide covers the networking, content delivery, and migration services essential for building secure, scalable, and high-performance generative AI architectures on AWS.[3][1][5][2][10][4][8]

[1](https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/)
[2](https://docs.aws.amazon.com/bedrock/latest/userguide/usingVPC.html)
[3](https://aws.amazon.com/datasync/)
[4](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/genrel05-bp01.html)
[5](https://aws.amazon.com/appsync/)
[6](https://docs.aws.amazon.com/datasync/latest/userguide/what-is-datasync.html)
[7](https://aws.amazon.com/blogs/storage/transferring-data-between-aws-accounts-using-aws-datasync/)
[8](https://aws.amazon.com/blogs/storage/automate-data-transfers-and-migrations-with-aws-datasync-and-terraform/)
[9](https://stackoverflow.com/questions/54691487/how-can-i-call-sagemaker-inference-endpoint-using-api-gateway)
[10](https://aws.amazon.com/about-aws/whats-new/2024/11/aws-appsync-ai-gateway-bedrock-integration-appsync-graphql/)
[11](https://awsfundamentals.com/blog/aws-edge-locations)
[12](https://stackoverflow.com/questions/55133263/is-aws-cloudfront-distribution-available-in-all-edge-locations)
[13](https://www.linkedin.com/posts/maishsk_secure-ingress-connectivity-to-amazon-bedrock-activity-7380527279401054208-g_mY)
[14](https://aws.amazon.com/blogs/machine-learning/creating-a-machine-learning-powered-rest-api-with-amazon-api-gateway-mapping-templates-and-amazon-sagemaker/)
[15](https://serverlessland.com/patterns/apigw-lambda-sagemaker-jumpstartendpoint-cdk-python)
[16](https://www.youtube.com/watch?v=Ol4JzIkeT4A)
[17](https://discuss.hashicorp.com/t/aws-sagemaker-runtime-integration/42322)
[18](https://www.w3schools.com/training/aws/aws-datasync-primer.php)
[19](https://www.datacamp.com/tutorial/aws-datasync)
[20](https://www.elastic.co/docs/explore-analyze/elastic-inference/inference-api)
