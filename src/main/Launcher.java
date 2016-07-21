package main;

// To add dlls to eclipse path: http://stackoverflow.com/questions/11123274/add-dll-to-java-library-path-in-eclipse-pydev-jython-project
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import jcuda.runtime.dim3;

// Important you need to compile the kernel separately from the command line
// http://www.jcuda.org/tutorial/TutorialIndex.html#CreatingKernels

// modify this for your CPU/GPU architecture:
// nvcc -cubin -m64 -arch sm_50 vectorQuery.cu -o vectorQuery.cubin

// We need CUBIN file so were not compiling during runtime!!
// Get compute capability of your GPU: https://developer.nvidia.com/cuda-gpus
public class Launcher {

	public static void main(String[] args) {
		// --------------------------------------------------------
		// Function Input Parameters
		int numRDFStatements = 50; // My bad GPU supports 107,374,182 triples
		
		// Dataset
		int[] subjects = new int[numRDFStatements];
		int[] predicates = new int[numRDFStatements];
		int[] objects = new int[numRDFStatements];
		int[] contexts = new int[numRDFStatements];
		int[] results = new int[numRDFStatements]; // 4 is a full match on the statement on the corresponding index
		
		// Query (Note GPU can't operate on null so encode wild card as -1)
		int subjectQuery = 0;
		int predicateQuery = 0;
		int objectQuery = 0;
		int contextQuery = 0;
		// --------------------------------------------------------
		
		/**
		 * Datastore Initialization
		 **/
		
		// Reset GPUs
		JCuda.cudaDeviceReset();
		
		// Determine the number of CUDA capable GPUs
		int[] numDevices = new int[1];
		JCuda.cudaGetDeviceCount(numDevices);
		System.out.println("Number of CUDA capable GPUs detected: "+numDevices[0]);
		if (numDevices[0] == jcuda.runtime.cudaError.cudaErrorNoDevice){
			System.err.println("GPU couldn't be found!");
			return;
		}
		
		// Choose a GPU to run on
		int deviceUsed = 0; // you can change this
		System.out.println("Running GPU Number: "+deviceUsed);
		JCuda.cudaSetDevice(deviceUsed);
		
		// Get GPU stats
		cudaDeviceProp prop = new cudaDeviceProp();
		JCuda.cudaGetDeviceProperties(prop, deviceUsed); // get stats for device 0
		System.out.println(prop.toString());
		
		// Check to ensure GPU can handle size of dataset 
		// 5 Arrays (subject<int>, predicate<int>, object<int>, context<int>, result<int>)
		int totalmem = 5 * (numRDFStatements*jcuda.Sizeof.INT);
		if (totalmem > prop.totalGlobalMem){
			System.err.println("GPU only supports "+ prop.totalGlobalMem+" bytes and this datastore requires "+ totalmem +" bytes!");
			return;
		}
				
		// Initialize array pointers
		CUdeviceptr  subjectArrayPointer = new CUdeviceptr ();
		CUdeviceptr  predicateArrayPointer = new CUdeviceptr ();
		CUdeviceptr  objectArrayPointer = new CUdeviceptr ();
		CUdeviceptr  contextArrayPointer = new CUdeviceptr ();
		CUdeviceptr  resultArrayPointer = new CUdeviceptr ();
		
		// Allocate arrays on GPU
		chckErr( JCuda.cudaMalloc(subjectArrayPointer, numRDFStatements*jcuda.Sizeof.INT) );
		chckErr( JCuda.cudaMalloc(predicateArrayPointer, numRDFStatements*jcuda.Sizeof.INT) );
		chckErr( JCuda.cudaMalloc(objectArrayPointer, numRDFStatements*jcuda.Sizeof.INT) );
		chckErr( JCuda.cudaMalloc(contextArrayPointer, numRDFStatements*jcuda.Sizeof.INT) );
		chckErr( JCuda.cudaMalloc(resultArrayPointer, numRDFStatements*jcuda.Sizeof.INT) );
		
		// Transfer data from CPU -> GPU
		chckErr( JCuda.cudaMemcpy(subjectArrayPointer, Pointer.to(subjects), numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice) );
		chckErr( JCuda.cudaMemcpy(predicateArrayPointer, Pointer.to(predicates), numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice) );
		chckErr( JCuda.cudaMemcpy(objectArrayPointer, Pointer.to(objects), numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice) );
		chckErr( JCuda.cudaMemcpy(contextArrayPointer, Pointer.to(contexts), numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice) );
				
		// Load the compiled Kernel onto GPU
		// This code below crashed my JVM :(
		CUmodule module = new CUmodule();
		chckErr( jcuda.driver.JCudaDriver.cuModuleLoad(module, "kernels\\vectorQuery.cubin") );
		CUfunction queryFunction = new CUfunction();
		chckErr( jcuda.driver.JCudaDriver.cuModuleGetFunction(queryFunction, module, "query") );
		
		// Calculating number of blocks and grids: http://www.resultsovercoffee.com/2011/02/cuda-blocks-and-grids.html
		int warpCount = (numRDFStatements / prop.warpSize) + (((numRDFStatements % prop.warpSize) == 0) ? 0 : 1);
		//System.out.println("warpCount: "+warpCount);
		
		int warpPerBlock = prop.maxThreadsPerBlock/prop.warpSize;
		//System.out.println("warpPerBlock: "+warpPerBlock);
		 
	    int blockCount = (warpCount / warpPerBlock) + (((warpCount % warpPerBlock) == 0) ? 0 : 1);
	    //System.out.println("blockCount: "+blockCount);
	    
	    dim3 BlockDim = new dim3(prop.maxThreadsPerBlock, 1, 1);
	    dim3 GridDim  = new dim3(blockCount, 1, 1);
	    System.out.println("BlockDim: "+BlockDim);
	    System.out.println("GridDim: "+GridDim);
	    
	    /**
		 * Running Querys
		 **/
	    
	    // Run Query Kernel Code on GPU
	    // ---------------------------------------------------------
	    // Havn't tested this! 
	   
	    // Setup kernel Parameters
		Pointer kernelParameters = Pointer.to(
				Pointer.to(new int[] { numRDFStatements }),
				Pointer.to(new int[] { subjectQuery }), 
				Pointer.to(new int[] { predicateQuery }), 
				Pointer.to(new int[] { objectQuery }), 
				Pointer.to(new int[] { contextQuery }), 
				Pointer.to(subjects),
				Pointer.to(predicates),
				Pointer.to(objects), 
				Pointer.to(contexts), 
				Pointer.to(results));

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
		//chckErr( JCuda.cudaMemcpy(Pointer.to(results), resultArrayPointer, numRDFStatements*jcuda.Sizeof.INT, jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost) );
			
		
		// Operate on results
		System.out.println("Results: ");
		for(int i = 0; i < numRDFStatements; i++ ){
			System.out.println(results[i]+", ");
		}
		
		/**
		 * Cleaning up GPU on close
		 **/
		chckErr( JCuda.cudaFree(subjectArrayPointer) );
		chckErr( JCuda.cudaFree(predicateArrayPointer) );
		chckErr( JCuda.cudaFree(objectArrayPointer) );
		chckErr( JCuda.cudaFree(contextArrayPointer) );
		chckErr( JCuda.cudaFree(resultArrayPointer) );
	}
	
	private static void chckErr(int error){
		if(error != jcuda.runtime.cudaError.cudaSuccess){
			System.err.println("CUDA Error: "+JCuda.cudaGetErrorString(error));
			 System.exit(1);
		}
	}
}
