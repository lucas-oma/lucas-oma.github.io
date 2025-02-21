interface Publication {
	  title: string
	  abstract: string
    keywords: string
    url: string
    publisher: string
}

interface ProfileData {
    name: string
    avatar: string
    bio: string
    resumeLink: string
    publications: Publication[]
}

export const profileData = {
  name: "Lucas O. Martinez",
  avatar: "https://media.licdn.com/dms/image/v2/D5635AQGzfVIIkQMZdQ/profile-framedphoto-shrink_200_200/profile-framedphoto-shrink_200_200/0/1734600491079?e=1740708000&v=beta&t=ku25OABzGcXZt4sRJ8PmFet4JoTpQWeZH79mLn6YG5M",
  resumeLink: "/media/pdf/Lucas_Martinez_Resume.pdf",
  publications: [
    {
      title: "On the Measurement of io_uring Performance",
      abstract: "A strategy for evaluating io_uring performance within Envoy Service Mesh.",
      keywords: "io_uring, Envoy Proxy, network performance",
      publisher: "CLEI Electronic Journal",
      url: "https://doi.org/10.19153/cleiej.27.2.3"
    },
    {
      title: "A Measurement Strategy for io_uring Performance",
      abstract: "A detailed measurement strategy for optimizing network efficiency.",
      keywords: "io_uring, performance, benchmarking",
      publisher: "IEEE",
      url: "https://doi.org/10.1109/CLEI60451.2023.10346093"
    }
  ],

  projects: [
    {
      title: "GradRag LLM",
      role: "Machine Learning Engineer",
      description: "A Retrieval-Augmented Generation (RAG) application for PhD aspirants to search professors using embeddings and vector-based similarity.",
      url: ""
    },
    {
      title: "Improving State Space Models for Computer Vision",
      role: "Machine Learning Research Engineer",
      description: "Enhanced state-space models for computer vision tasks using optimization techniques for Vision Mamba and MambaVision architectures.",
      url: "https://github.com/rdaggs/vim_ssm_cv24"
    },
    {
      title: "Hedge Fund Alpha: Clustering Stocks for Portfolio Enhancement",
      role: "Quant Research Engineer",
      description: "Implemented a predictive model to classify sector-based stock clusters, leveraging alternative data sources to enhance investment strategies.",
      url: ""
    }
  ]
}
