MGARDPlug operator developed to compress and decompress arbitrary unstructured meshed data. This operator is part of ADIOS's plug library, which will perform compression/decompression inside ADIOS's PUT and GET operation. 

Once installed, this operator can be used through "var.AddOperation("plugin", params);".

This operator requires users to prepare a mesh2grid mapping file, and provide it through the params["meshfile"]. For more detailed examples, please refer to the Unstructured-ReMesh/mgardPlug_adios_ge.cpp and mgardPlug_adios_decompress.cpp.  
