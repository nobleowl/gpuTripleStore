package main;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.dim3;

public class gpuTriplestore {
	private int numRDFStatements; // My bad GPU supports 107,374,182 triples
	
	private String[] subjects;
	private String[] predicates;
	private String[] objects;
	private String[] contexts;
	private int[] results;
	
	private cudaDeviceProp prop;
	private CUfunction queryFunction;
	
	private CUdeviceptr subjectArrayPointer;
	private CUdeviceptr predicateArrayPointer;
	private CUdeviceptr objectArrayPointer;
	private CUdeviceptr contextArrayPointer;
	private CUdeviceptr resultArrayPointer;
	
	private dim3 BlockDim;
	private dim3 GridDim;
		
	public gpuTriplestore(int deviceNumber){
		// Reset GPUs
		JCuda.cudaDeviceReset();

		// Determine the number of CUDA capable GPUs
		int[] numDevices = new int[1];
		JCuda.cudaGetDeviceCount(numDevices);
		System.out.println("Number of CUDA capable GPUs detected: " + numDevices[0]);
		if (numDevices[0] == jcuda.runtime.cudaError.cudaErrorNoDevice) {
			System.err.println("GPU couldn't be found!");
			return;
		}

		// Choose a GPU to run on
		System.out.println("Running GPU Number: " + deviceNumber);
		// Initialize the driver and create a context for the first device.
		cuInit(deviceNumber);
		CUcontext pctx = new CUcontext();
		CUdevice dev = new CUdevice();
		cuDeviceGet(dev, deviceNumber);
		cuCtxCreate(pctx, deviceNumber, dev);
		
		// Get GPU stats
		prop = new cudaDeviceProp();
		JCuda.cudaGetDeviceProperties(prop, deviceNumber); // get stats for device
		System.out.println(prop.toString());

		// Initialize array pointers
		subjectArrayPointer = new CUdeviceptr();
		predicateArrayPointer = new CUdeviceptr();
		objectArrayPointer = new CUdeviceptr();
		contextArrayPointer = new CUdeviceptr();
		resultArrayPointer = new CUdeviceptr();

		// Load the compiled Kernel onto GPU
		try {
			prepareCubinFile("kernels\\vectorQuery.cu");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("compiled!");
		CUmodule module = new CUmodule();
		chckErr(jcuda.driver.JCudaDriver.cuModuleLoad(module, "kernels\\vectorQuery.cubin"));
		queryFunction = new CUfunction();
		chckErr(jcuda.driver.JCudaDriver.cuModuleGetFunction(queryFunction, module, "query"));
	}
	
	public void loadData(String[] subject, String[] predicate, String[] object, String[] context ){
		// Load copy of dataset into java
		subjects = subject;
		predicates = predicate;
		objects = object;
		contexts = context;
		numRDFStatements = subjects.length;
		
		// Hash the datasets
		int[] subjectHash = HashArray(subjects);
		int[] predicateHash = HashArray(predicates);
		int[] objectHash = HashArray(objects);
		int[] contextHash = HashArray(contexts);
		
		// Load Data onto GUP
		chckErr( JCuda.cudaMemcpy(subjectArrayPointer, Pointer.to(subjectHash), numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice) );
		chckErr( JCuda.cudaMemcpy(predicateArrayPointer, Pointer.to(predicateHash), numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice) );
		chckErr( JCuda.cudaMemcpy(objectArrayPointer, Pointer.to(objectHash), numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice) );
		chckErr( JCuda.cudaMemcpy(contextArrayPointer, Pointer.to(contextHash), numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice) );
		
		// Calculating number of blocks and grids:
		// http://www.resultsovercoffee.com/2011/02/cuda-blocks-and-grids.html
		int warpCount = (numRDFStatements / prop.warpSize) + (((numRDFStatements % prop.warpSize) == 0) ? 0 : 1);
		// System.out.println("warpCount: "+warpCount);

		int warpPerBlock = prop.maxThreadsPerBlock / prop.warpSize;
		// System.out.println("warpPerBlock: "+warpPerBlock);

		int blockCount = (warpCount / warpPerBlock) + (((warpCount % warpPerBlock) == 0) ? 0 : 1);
		// System.out.println("blockCount: "+blockCount);

		BlockDim = new dim3(prop.maxThreadsPerBlock, 1, 1);
		GridDim = new dim3(blockCount, 1, 1);
		System.out.println("BlockDim: " + BlockDim);
		System.out.println("GridDim: " + GridDim);
		
		// intialize output array
		results = new int[numRDFStatements];
	}
	
	// Only supports 1 variable in a bgp
	public ArrayList<String> query(String subject, String predicate, String object, String context){
		int var = 0;
		int subjectQuery = -1;
		if( subject.charAt(0) != '?'){
			subjectQuery = subject.hashCode();
		}
		
		int predicateQuery = -1;
		if( predicate.charAt(0) != '?'){
			predicateQuery = predicate.hashCode();
			var = 1;
		}
		
		int objectQuery = -1;
		if( object.charAt(0) != '?'){
			objectQuery = object.hashCode();
			var = 2;
		}
		
		int contextQuery = -1;
		if( context.charAt(0) != '?'){
			contextQuery = context.hashCode();
			var = 3;
		}
		
		
		// Setup kernel Parameters
		Pointer kernelParameters = Pointer.to(
				Pointer.to(new int[] { numRDFStatements }),
				Pointer.to(new int[] { subjectQuery }), 
				Pointer.to(new int[] { predicateQuery }), 
				Pointer.to(new int[] { objectQuery }), 
				Pointer.to(new int[] { contextQuery }), 
				Pointer.to(subjectArrayPointer),
				Pointer.to(predicateArrayPointer),
				Pointer.to(objectArrayPointer), 
				Pointer.to(contextArrayPointer), 
				Pointer.to(resultArrayPointer));

		// Call the kernel function
		int errorCode = jcuda.driver.JCudaDriver.cuLaunchKernel(queryFunction, 
			GridDim.x, 1, 1, // Grid dimension
			BlockDim.x, 1, 1, // Block dimension
			0, null, // Shared memory size and stream
			kernelParameters, null // Kernel- and extra parameters
		);
				
		// check for errors
		chckErr( errorCode );
		// ---------------------------------------------------------
				
		// Copy Results back from GPU -> CPU
		// Note some times error codes are delayed, so errors here may be from kernel
		chckErr( JCuda.cudaMemcpy(Pointer.to(results), resultArrayPointer, numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost) );
				
		ArrayList<String> resultOut = new ArrayList<String>((3*numRDFStatements)/4);
		
		// Operate on results
		for(int i = 0; i < numRDFStatements; i++ ){
			if( results[i] == 4 ){
				switch(var){
				case 0:
					resultOut.add(subjects[i]);
					break;
				case 1:
					resultOut.add(predicates[i]);
					break;
				case 2:
					resultOut.add(objects[i]);
					break;
				default:
					resultOut.add(contexts[i]);
				}
			}
		}
		
		return resultOut;
	}
	
	private int[] HashArray(String[] array) {
		int[] hash= new int[array.length];
		for(int i =0; i < array.length; i++){
			hash[i] = array[i].hashCode();
		}
		return hash;
	}

	private static void chckErr(int error){
		if(error != jcuda.runtime.cudaError.cudaSuccess){
			System.err.println("CUDA Error: "+JCuda.cudaGetErrorString(error));
			 System.exit(1);
		}
	}
	
	 /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is 
     * compiled from the given file using NVCC. The name of the 
     * PTX file is returned. 
     * 
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static String prepareCubinFile(String cuFileName) throws IOException {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"cubin";
        File ptxFile = new File(ptxFileName);
        /*
        if (ptxFile.exists())
        {
            return ptxFileName;
        }
        */
        
        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");        
        String command = 
            "nvcc " + modelString + " -arch sm_50" + " -cubin "+
            cuFile.getPath()+" -o "+ptxFileName;
        
        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage = 
            new String(toByteArray(process.getErrorStream()));
        String outputMessage = 
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }
        
        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }
    
    /**
     * Fully reads the given InputStream and returns it as a byte array
     *  
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream) 
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
}

