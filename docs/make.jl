using ISODATA
using Documenter

makedocs(;
    modules=[ISODATA],
    authors="Noah Gardner <ngngardner@gmail.com> and contributors",
    repo="https://github.com/ngngardner/ISODATA.jl/blob/{commit}{path}#L{line}",
    sitename="ISODATA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ngngardner.github.io/ISODATA.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ngngardner/ISODATA.jl",
)
