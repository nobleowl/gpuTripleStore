extern "C"
__global__ void query(int size, int subjectQuery, int predicateQuery, int objectQuery, int contextQuery, int *subjects, int *predicates, int *objects, int *contexts, int *result)
{
	// Get thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Don't operate on memory outside of array
    if (i<size){
    
    	// Add 1 to the result if subject, predicate, object, or context are a match or if they are a wild card (variable)
    	int accumulator = 0;
    	
    	// subject
    	accumulator += ((subjectQuery == subjects[i]) ? 1 : 0);
    	accumulator += ((subjectQuery == -1) ? 1 : 0);
    	
    	// predicate
    	accumulator += ((predicateQuery == predicates[i]) ? 1 : 0);
    	accumulator += ((predicateQuery == -1) ? 1 : 0);
    	
    	// object
    	accumulator += ((objectQuery == objects[i]) ? 1 : 0);
    	accumulator += ((objectQuery == -1) ? 1 : 0);
    	
    	// context
    	accumulator += ((contextQuery == contexts[i]) ? 1 : 0);
    	accumulator += ((contextQuery == -1) ? 1 : 0);
    	
    	result[i] = accumulator;
    }
}