interface Publication {
	  title: string
	  abstract: string
    keywords: string
    url: string
    publisher: string
}

interface Projects {
  title: string
  description: string
  keywords: string
  pdf: string | undefined
  github: string | undefined
  website: string | undefined
  date: string
  image: string | undefined
}

interface ProfileData {
    name: string
    avatar: string
    // bio: string
    resumeLink: string
    publications: Publication[]
    projects: Projects[]
}

export const profileData: ProfileData = {
  name: "Lucas O. Martinez",
  avatar: "/media/images/profile-pic.jpg",
  resumeLink: "/media/pdf/Lucas_Martinez_Resume.pdf",
  publications: [
    {
      title: "On the Measurement of io_uring Performance",
      abstract: "A growing number of projects have adopted io_uring to reduce system calls and enhance network performance. This technique holds promise for modern infrastructure, including service meshes. Envoy Proxy has been integrating io_uring, with early trials showing a 10% bandwidth increase and 10% latency reduction, though lacking reproducibility data. This study systematically evaluates io_uring's impact on an Envoy-based service mesh, using a structured matrix of operations and protocols. Our results confirm latency reductions and bandwidth gains, aligning with initial claims for HTTP/1 and exceeding them for HTTP/2.",
      keywords: "io_uring, Envoy Proxy, network performance",
      publisher: "CLEI Electronic Journal",
      url: "https://doi.org/10.19153/cleiej.27.2.3"
    },
    {
      title: "A Measurement Strategy for io_uring Performance",
      abstract: "Over the past years, the service mesh paradigm has gained traction, solidifying its utility in modern infrastructures. Additionally, numerous projects in the field have recently incorporated support for io_uring — a technique to avoid system calls — affirming the trend in exploring this new approach to accelerate processing capabilities in network services. Within this context, Envoy is also in the process of gaining io_uring support. This work proposes the exploration of metrics to measure the impact of io_uring, which avoids the cost of utilizing resource-intensive system calls in an Envoy-based service mesh. Alongside a proposed measurement strategy, the design and implementation of a testing environment are presented. Test results showcase improvements in both latency performance and bandwidth utilization efficiency.",
      keywords: "io_uring, performance, benchmarking",
      publisher: "IEEE",
      url: "https://doi.org/10.1109/CLEI60451.2023.10346093"
    }
  ],

  projects: [
    {
      title: "Fantastic Router",
      description: "A natural-language routing engine that allows users to navigate complex applications using plain queries like “show me James Smith's monthly income.” Maps free-form text directly to structured routes (e.g., /landlords/james-smith-123/financials). Designed with extensibility in mind, supporting pluggable adapters for databases, LLM providers, and vector stores, and built-in role-based access control (RBAC).",
      keywords: "Natural Language Interface, Routing Engine, LLM Integration, Vector Search, Extensible Framework, RBAC",
      pdf: undefined,
      github: "https://github.com/lucas-oma/fantastic-router",
      website: undefined,
      date: "2025-08-15",
      image: "/media/images/projects/fantastic-router.png"
    },
    {
      title: "VectorDB from Scratch",
      description: "A from-scratch vector database with a REST API and MongoDB persistence. Supports exact (Flat) and approximate indexes (IVF, LSH SimHash), k-NN search, document/chunk management, and Postman-tested workflows.",
      keywords: "Vector Database, ANN, IVF, LSH, Similarity Search, k-NN, Embeddings, FastAPI, MongoDB",
      pdf: undefined,
      github: "https://github.com/lucas-oma/vectorDB_from_scratch",
      website: undefined,
      date: "2025-08-19",
      image: "/media/images/projects/vectordbfromscratch.png"
    },
    {
      title: "Propi - Property Management & Financial System",
      description: "A comprehensive React/TypeScript SaaS platform for property management with automated invoice processing, AI-powered document analysis using Mistral OCR and Gemini AI, real-time payment tracking, and integrated WhatsApp notifications. Features multi-currency support, tax calculations, and automated tenant communication workflows.",
      keywords: "Property Management, AI Document Processing, Mistral OCR, Gemini AI, React TypeScript, Supabase, WhatsApp Integration, Multi-currency, Tax Calculations, SaaS",
      pdf: undefined,
      github: undefined,
      website: "https://www.getpropi.com",
      date: "2025-05-20",
      image: "/media/images/projects/propi.png"
    },
    {
      title: "Duplicate Job-posts Detection System",
      description: "A real-time FastAPI system to detect semantically similar job posts using SentenceTransformers for embeddings and Milvus for vector similarity search. Includes threshold tuning, evaluation tools, and scalable batch processing.",
      keywords: "Duplicate Detection, SentenceTransformers, Milvus, DiskANN, FastAPI, Vector Search, Embeddings",
      pdf: undefined,
      github: "https://github.com/lucas-oma/duplicated-jobs-detection",
      website: undefined,
      date: "2025-04-14",
      image: "/media/images/projects/duplicatejobsdetection.png"
    },
    {
      title: "JobRAG",
      description: "A full-stack Retrieval-Augmented Generation (RAG) platform that helps users discover job opportunities by querying vector-embedded job descriptions stored in Qdrant. Includes a scraping pipeline, LLM-powered description extraction, and a modern Next.js frontend.",
      keywords: "LLM, RAG, Qdrant, FastAPI, Next.js",
      pdf: undefined,
      github: undefined,
      website: "https://job-rag.vercel.app",
      date: "2025-04-11",
      image: "/media/images/projects/jobrag.png"
    },
    {
      title: "GradRag LLM",
      description: "A Retrieval-Augmented Generation (RAG) application for PhD aspirants to search professors using embeddings and vector-based similarity.",
      keywords: "LLM, RAG, Qdrant",
      pdf: undefined,
      github: undefined,
      website: undefined,
      date: "2025-01-10",
      image: "/media/images/projects/gradrag.png"
    },
    {
      title: "Improving State Space Models for Computer Vision",
      description: "Enhanced state-space models for computer vision tasks using optimization techniques for Vision Mamba and MambaVision architectures.",
      keywords: "State Space Models, Vision Mamba, MambaVision, Model Optimization",
      pdf: "/media/pdf/Improving_State_Space_Models_for_Computer_Vision.pdf",
      github: "https://github.com/rdaggs/vim_ssm_cv24",
      website: undefined,
      date: "2024-12-20",
      image: "/media/images/projects/mamba.png"
    },
    {
      title: "HiveCluster: Strategic Stock Portfolio Construction",
      description: "Implemented a predictive model to classify sector-based stock clusters, leveraging alternative data sources to enhance investment strategies.",
      keywords: "Predictive Analytics, Machine Learning, Stock Market, Market Leaders, Portfolio, Hedge Fund Alpha, Lead-lag Relationship, Clustering",
      pdf: "/media/pdf/HiveCluster-Strategic_Stock_Portfolio_Construction.pdf",
      github: undefined,
      website: undefined,
      date: "2024-12-24",
      image: "/media/images/projects/hivecluster.png"
    },
    {
      title: "Regression with Text Features: an Experimental Approach Evaluated on Real-World Applications",
      description: "Investigated traditional and embedding-based regression techniques on Supreme Court decisions and LinkedIn job descriptions.",
      keywords: "Text regression, Embeddings, TF-IDF, SVR, MLP, LLM, BERT",
      pdf: "/media/pdf/Regression_with_Text_Features.pdf",
      github: "https://github.com/apeterson7/cs2565-project",
      website: undefined,
      date: "2024-12-13",
      image: "/media/images/projects/textregression.png"
    },
    {
      title: "Exploring NeRF: Project Report",
      description: "Experimented with NeRF improvements through architectural changes, loss functions, and contrast adjustments for better 3D scene reconstruction.",
      keywords: "NeRF, 3D Reconstruction, Deep Learning, Computer Vision, Model Optimization",
      pdf: "/media/pdf/Exploring_NeRF-report.pdf",
      github: "https://github.com/lucas-oma/exploring_NeRF",
      website: undefined,
      date: "2024-05-09",
      image: "/media/images/projects/nerf.png"
    },
    {
      title: "Hyperparameter tuning on LaVIN, an Efficient Vision-Language Instruction Tuning for Large Language Models",
      description: "Extended the capabilities of large vision-language models by fine-tuning LaVIN with different learning rate schedulers and dropout strategies.",
      keywords: "Vision-language Models, Multimodal AI, LaVIN, LLaMA, Fine-tuning, LLM",
      pdf: undefined,
      github: "https://github.com/HyejunShin/LaVIN",
      website: undefined,
      date: "2023-12-20",
      image: "/media/images/projects/lavin.png"
    }
  ]
}
