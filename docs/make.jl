using FiniteMPSAlgorithms
using Documenter

DocMeta.setdocmeta!(FiniteMPSAlgorithms, :DocTestSetup, :(using FiniteMPSAlgorithms); recursive=true)

makedocs(;
    modules=[FiniteMPSAlgorithms],
    authors="Guochu",
    sitename="FiniteMPSAlgorithms",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://guochu.github.io/FiniteMPSAlgorithms.jl",
    ),
    pages=[
        "Home" => "index.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/guochu/FiniteMPSAlgorithms.jl",
)
