#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define TILE_DIM 32
#define BLOCK_SIZE (TILE_DIM * TILE_DIM)
#define SCAN_BLOCK_DIM BLOCK_SIZE

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

__constant__ GlobalConstants cuConstRendererParams;

__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];

// External device helpers from starter code
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"
#include "exclusiveScan.cu_inl"


__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159f;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) {
        return;
    }

    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i + 1];

    position[index3j] += velocity[index3j] * dt;
    position[index3j + 1] += velocity[index3j + 1] * dt;

    float sx = position[index3j];
    float sy = position[index3j + 1];

    float cxsx = sx - cx;
    float cysy = sy - cy;

    float dist = sqrtf(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) {
        float angle = (sfIdx * 2.f * pi) / NUM_SPARKS;
        float sinA = sinf(angle);
        float cosA = cosf(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j + 1] = position[index3i + 1] + y;
        position[index3j + 2] = 0.0f;

        velocity[index3j] = cosA / 5.0f;
        velocity[index3j + 1] = sinA / 5.0f;
        velocity[index3j + 2] = 0.0f;
    }
}

__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}

__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f;
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    float oldVelocity = velocity[index3 + 1];
    float oldPosition = position[index3 + 1];

    if (oldVelocity == 0.f && oldPosition == 0.f) {
        return;
    }

    if (position[index3 + 1] < 0 && oldVelocity < 0.f) {
        velocity[index3 + 1] *= kDragCoeff;
    }

    velocity[index3 + 1] += kGravity * dt;
    position[index3 + 1] += velocity[index3 + 1] * dt;

    if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3 + 1] - oldPosition) < epsilon) {
        velocity[index3 + 1] = 0.f;
        position[index3 + 1] = 0.f;
    }
}

__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f;
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    float forceScaling = fminf(fmaxf(1.f - position.z, .1f), 1.f);

    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    if ((position.y + radius < 0.f) ||
        (position.x + radius) < -0.f ||
        (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// Register-accumulating pixel shader reading circle attributes from shared memory
template <bool isSnowflake>
__device__ __inline__ void
shadePixelTiled(float2 pixelCenter, float3 p, float rad, float3 cachedRgb, float4& accumColor) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;
    float maxDist = rad * rad;

    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    if (isSnowflake) {
        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrtf(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f - p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
        alpha = maxAlpha * expf(-1.f * falloffScale * normPixelDist * normPixelDist);
    } else {
        rgb = cachedRgb;
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;
    accumColor.x = alpha * rgb.x + oneMinusAlpha * accumColor.x;
    accumColor.y = alpha * rgb.y + oneMinusAlpha * accumColor.y;
    accumColor.z = alpha * rgb.z + oneMinusAlpha * accumColor.z;
    accumColor.w = alpha + accumColor.w;
}

// Tiled renderer: one 32x32 block (1024 threads) per 32x32 pixel tile
template <bool isSnowflake>
__global__ void kernelRenderCirclesTiled() {

    __shared__ uint scanInput[BLOCK_SIZE];
    __shared__ uint scanOutput[BLOCK_SIZE];
    __shared__ uint scanScratch[2 * BLOCK_SIZE];
    __shared__ int  tileCircles[BLOCK_SIZE];

    __shared__ float3 sharedPos[BLOCK_SIZE];
    __shared__ float  sharedRad[BLOCK_SIZE];
    __shared__ float3 sharedCol[BLOCK_SIZE];

    int linearTid = threadIdx.y * TILE_DIM + threadIdx.x;
    int pixelX = blockIdx.x * TILE_DIM + threadIdx.x;
    int pixelY = blockIdx.y * TILE_DIM + threadIdx.y;

    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // Normalized [0, 1] bounding box of this tile
    int tileStartX = blockIdx.x * TILE_DIM;
    int tileStartY = blockIdx.y * TILE_DIM;
    int tileEndX = min(tileStartX + TILE_DIM, imageWidth);
    int tileEndY = min(tileStartY + TILE_DIM, imageHeight);

    float boxL = tileStartX * invWidth;
    float boxR = tileEndX * invWidth;
    float boxB = tileStartY * invHeight;
    float boxT = tileEndY * invHeight;

    bool inBounds = (pixelX < imageWidth && pixelY < imageHeight);
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));

    int pixelOffset = 4 * (pixelY * imageWidth + pixelX);
    float4 accumColor = inBounds
        ? *(float4*)(&cuConstRendererParams.imageData[pixelOffset])
        : make_float4(0.f, 0.f, 0.f, 0.f);

    int numCircles = cuConstRendererParams.numCircles;

    for (int chunkStart = 0; chunkStart < numCircles; chunkStart += BLOCK_SIZE) {
        int circleIdx = chunkStart + linearTid;
        uint overlaps = 0;

        if (circleIdx < numCircles) {
            float3 p = *(float3*)(&cuConstRendererParams.position[3 * circleIdx]);
            float rad = cuConstRendererParams.radius[circleIdx];

            // Note: circleBoxTest.cu_inl expects (boxL, boxR, boxT, boxB)
            if (circleInBoxConservative(p.x, p.y, rad, boxL, boxR, boxT, boxB)) {
                if (circleInBox(p.x, p.y, rad, boxL, boxR, boxT, boxB)) {
                    overlaps = 1;
                    sharedPos[linearTid] = p;
                    sharedRad[linearTid] = rad;
                    if (!isSnowflake) {
                        sharedCol[linearTid] = *(float3*)(&cuConstRendererParams.color[3 * circleIdx]);
                    }
                }
            }
        }

        // Fast-path: if zero circles in this 1024-circle chunk intersect the tile, skip scan and shading
        if (!__syncthreads_or(overlaps)) {
            continue;
        }

        scanInput[linearTid] = overlaps;
        __syncthreads();

        // Exclusive prefix sum across the 1024 flags
        sharedMemExclusiveScan(linearTid, scanInput, scanOutput, scanScratch, BLOCK_SIZE);
        __syncthreads();

        int numInTile = scanOutput[BLOCK_SIZE - 1] + scanInput[BLOCK_SIZE - 1];
        if (overlaps) {
            tileCircles[scanOutput[linearTid]] = linearTid;
        }
        __syncthreads();

        // Walk matching circles in strict ascending index order
        if (inBounds) {
            for (int i = 0; i < numInTile; i++) {
                int localIdx = tileCircles[i];
                shadePixelTiled<isSnowflake>(
                    pixelCenterNorm,
                    sharedPos[localIdx],
                    sharedRad[localIdx],
                    sharedCol[localIdx],
                    accumColor);
            }
        }

        // Ensure all threads finish reading shared arrays before next chunk overwrites them
        __syncthreads();
    }

    if (inBounds) {
        *(float4*)(&cuConstRendererParams.imageData[pixelOffset]) = accumColor;
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene, int seed) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius, seed);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);
}

void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

void
CudaRenderer::clearImage() {

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::advanceAnimation() {
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {

    dim3 blockDim(TILE_DIM, TILE_DIM, 1);
    dim3 gridDim(
        (image->width + TILE_DIM - 1) / TILE_DIM,
        (image->height + TILE_DIM - 1) / TILE_DIM);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelRenderCirclesTiled<true><<<gridDim, blockDim>>>();
    } else {
        kernelRenderCirclesTiled<false><<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}