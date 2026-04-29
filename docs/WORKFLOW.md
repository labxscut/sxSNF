# sxSNF Overall Workflow

This page is the high-level workflow entry for the repository and pydoc documentation.

## Pipeline Diagram

```mermaid
flowchart TD
    start[Start] --> parseArgs[ParseCLIArgs]
    parseArgs --> dataChoice{DataSource}
    dataChoice -->|Simulated| genData[GenerateSimulatedData]
    dataChoice -->|Real| loadData[LoadRealMultiOmicsData]
    genData --> preprocess[NormalizeEachModality]
    loadData --> preprocess
    preprocess --> buildGraphs[BuildSimilarityGraphsPerModality]
    buildGraphs --> snfFusion[RunSNFFusion]
    snfFusion --> fusedGraph[FusedSimilarityNetwork]
    fusedGraph --> initGNN[InitializeGNNModel]
    initGNN --> trainGNN[TrainGNNEncoder]
    trainGNN --> embeddings[GetCellEmbeddings]
    embeddings --> cluster[RunKMeansClustering]
    embeddings --> vizEmb[VisualizeEmbeddings]
    cluster --> metrics[ComputeNMIandARI]
    cluster --> vizCluster[VisualizeClusterResults]
    metrics --> saveOutputs[SaveEmbeddingsLabelsMetrics]
    vizEmb --> saveOutputs
    vizCluster --> saveOutputs
    saveOutputs --> endNode[End]
```

## Search Keywords

sxSNF, SNF, GNN, workflow, pipeline, multi-omics, similarity graph, fusion, embeddings, clustering, NMI, ARI, pydoc.
