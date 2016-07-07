#
@everywhere include("KronGraph500NoPerm.jl")
@everywhere include("StrFileWrite.jl")
@everywhere include("StrFileRead.jl")

function PageRankPipeline(SCALE,EdgesPerVertex,Nfile);

  Nmax = 2.^SCALE;                           # Max vertex ID.
  M = EdgesPerVertex .* Nmax;                # Total number of edges.
  myFiles = collect(1:Nfile).';              # Set list of files.
  #
  tab = Char(9)
  nl = Char(10)
  Niter = 20                                      # Number of PageRank iterations.
  c = 0.15                                        # PageRank damping factor.

  println("Number of Edges: " * string(M) * ", Maximum Possible Vertex: " * string(Nmax));


  ########################################################
  # Kernel 0: Generate a Graph500 Kronecker graph and save to data files.
  ########################################################
  println("Kernel 0: Generate Graph, Write Edges");
  tic();
  let SCALE=SCALE,EdgesPerVertex=EdgesPerVertex,Nfile=Nfile
    pmap(i -> begin
      fname = "data/K0/" * string(i) * ".tsv";
      println("  Writing: " * fname);                          # Read filename.
      srand(i);                                                # Set random seed to be unique for this file.
      ut, vt = KronGraph500NoPerm(SCALE,EdgesPerVertex./Nfile);  # Generate data.
      writeuv(fname, ut, vt)
      nothing
    end, myFiles)
  end
  K0time = toq();
  println("K0 Time: " * string(K0time) * ", Edges/sec: " * string(M./K0time));

  ########################################################
  # Kernel 1: Read data, sort data, and save to files.
  ########################################################
  println("Kernel 1: Read, Sort, Write Edges");
  tic();

  # Each worker sorts only a static range of Nmax. TODO: Currently unbalanced, balance it out.
    u=SharedArray(Int, M)
    v=SharedArray(Int, M)

    # Read in all the files into one array.
    nPerFile = div(M,Nfile)
    pmap(i->begin
      fname = "data/K0/" * string(i) * ".tsv"
      println("  Reading: " * fname);  # Read filename.
      ut,vt = StrFileRead(fname)
      # Concatenate to u,v
      startOffset = (i-1)*nPerFile + 1
      endOffSet = i*nPerFile

      u[startOffset:endOffSet] = ut
      v[startOffset:endOffSet] = vt
      nothing
    end, myFiles)

    rangeInWorker = div(Nmax,nworkers())
    localsort(i) = begin
      lb = (i-1)*rangeInWorker + 1
      ub = i*rangeInWorker
      idxs = find(x->(x >= lb && x <= ub), u)   # To be sorted on this worker

      lt = (x,y) -> (u[x] < u[y])
      sorted = sort(idxs; lt=lt)

      f=Future()
      put!(f, sorted)
      (f, length(sorted))
    end

    u2=SharedArray(Int, M)
    v2=SharedArray(Int, M)
    refs = asyncmap((i,p)->remotecall_fetch(localsort, p, i), 1:nworkers(), workers())
    offset=1

    for (f,l) in refs
      let offset=offset
        remotecall_fetch((f, idx,l) -> begin
              u2[idx:idx+l-1] = u[fetch(f)]
              v2[idx:idx+l-1] = v[fetch(f)]
              nothing
            end, f.where, f, offset, l)
      end
      offset += l
    end

    u = u2
    v = v2


#    sortIndex = sortperm(u)                      # Sort starting vertices.
#    u = u[sortIndex]                                  # Get starting vertices.
#    v = v[sortIndex]                                  # Get ending vertices.

  K1time1 = toq();
  tic();
    # Write all the data to files.
    c = size(u,1)/length(myFiles)        # Compute first edge of file.
    pmap(i -> begin
      jEdgeStart = round(Int, (i-1)*c+1)# Compute first edge of file.
      jEdgeEnd = round(Int, i*c)          # Compute last edge of file.
      uu = sub(u,jEdgeStart:jEdgeEnd)                                 # Select start vertices.
      vv = sub(v,jEdgeStart:jEdgeEnd)                                 # Select end vertices.
      fname = "data/K1/" * string(i) * ".tsv"
      println("  Writing: " * fname)                              # Create filename.
      writeuv(fname, uu, vv)
    end, myFiles)

  K1time2 = toq();
  K1time = K1time1 + K1time2;
  println("K1 Time (reading):" * string(K1time1) * ", Edges/sec: " * string(M./K1time1));
  println("K1 Time (writing):" * string(K1time2) * ", Edges/sec: " * string(M./K1time1));
  println("K1 Time: " * string(K1time) * ", Edges/sec: " * string(M./K1time));

exit()

  ########################################################
  # Kernel 2: Read data, filter data.
  ########################################################
  println("Kernel 2: Read, Filter Edges");
  tic();
    # Read in all the files into one array.
    for i in myFiles
      fname = "data/K1/" * string(i) * ".tsv";
      println("  Reading: " * fname);                # Read filename.
      ut,vt = StrFileRead(fname);
      if i == 1
         u = ut; v = vt;                             # Initialize starting and ending vertices.
      else
         append!(u, ut)
         append!(v, vt)
         # Get the rest of starting and ending vertices.
      end
    end

    # Construct adjacency matrix.
    A = sparse(vec(u),vec(v),1.0,Nmax,Nmax)      # Create adjacency matrix.

    # Filter and weight the adjacency matrix.
    din = sum(A,1)                               # Compute in degree.
    A[find(din == maximum(din))]=0               # Eliminate the super-node.
    A[find(din == 1)]=0                          # Eliminate the leaf-node.
    dout = sum(A,2)                              # Compute the out degree.
    is = find(dout)                               # Find vertices with outgoing edges (dout > 0).
    DoutInvD = zeros(Nmax)        # Create diagonal weight matrix.
    DoutInvD[is] = 1./dout[is]
    scale!(DoutInvD, A)           # Apply weight matrix.
  K2time = toq();
  println("K2 Time: " * string(K2time) * ", Edges/sec: " * string(M./K2time));


  ########################################################
  # Kernel 3: Compute PageRank.
  ########################################################
  println("Kernel 3: PageRank");
  tic();

    r = rand(1,Nmax);                     # Generate a random starting rank.
    r = r ./ norm(r,1);                   # Normalize
    a = (1-c) ./ Nmax;                    # Create damping vector

    for i=1:Niter
        s = r * A
        scale!(s, c)
        r = s .+ (a * sum(r,2));                # Compute PageRank.
    end

    r=r./norm(r,1);

  K3time = toq();
  println("  Sum of PageRank: " * string(sum(r)) );     # Force all computations to occur.
  println("K3 Time: " * string(K3time) * ", Edges/sec: " * string(Niter.*M./K3time));

  return K0time,K1time,K2time,K3time

end

########################################################
# PageRank Pipeline Benchmark
# Architect: Dr. Jeremy Kepner (kepner@ll.mit.edu)
# Julia Translation: Dr. Chansup Byun (cbyun@ll.mit.edu)
# MIT
########################################################
# (c) <2015> Massachusetts Institute of Technology
########################################################
