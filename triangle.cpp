/*
* Vulkan Example - Basic indexed triangle rendering
*
* Note:
*	This is a "pedal to the metal" example to show off how to get Vulkan up an displaying something
*	Contrary to the other examples, this one won't make use of helper functions or initializers
*	Except in a few cases (swap chain setup e.g.)
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/




#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream>
#include <vector>
#include <exception>

#include <random>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VulkanDevice.hpp"
#include "VulkanBuffer.hpp"
#include "VulkanModel.hpp"

#include "../external/stb/stb_font_consolas_24_latin1.inl"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Max. number of chars the text overlay buffer can hold
#define TEXTOVERLAY_MAX_CHAR_COUNT 2048

#include <math.h>

// Set to "true" to enable Vulkan's validation layers (see vulkandebug.cpp for details)
#define ENABLE_VALIDATION false

#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

/*
Mostly self-contained text overlay class
*/
class TextOverlay
{
private:
  vks::VulkanDevice *_vulkanDevice;

  VkQueue _queue;
  VkFormat _colorFormat;
  VkFormat _depthFormat;

  uint32_t *_frameBufferWidth;
  uint32_t *_frameBufferHeight;

  VkSampler _sampler;
  VkImage _image;
  VkImageView _view;
  VkBuffer _buffer;
  VkDeviceMemory _memory;
  VkDeviceMemory _imageMemory;
  VkDescriptorPool _descriptorPool;
  VkDescriptorSetLayout _descriptorSetLayout;
  VkDescriptorSet _descriptorSet;
  VkPipelineLayout _pipelineLayout;
  VkPipelineCache _pipelineCache;
  VkPipeline _pipeline;
  VkRenderPass _renderPass;
  VkCommandPool _commandPool;

  std::vector<VkFramebuffer*> _frameBuffers;
  std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;

  // Pointer to mapped vertex buffer
  glm::vec4 *_mapped = nullptr;

  stb_fontchar _stbFontData[STB_FONT_consolas_24_latin1_NUM_CHARS];
  uint32_t _numLetters;
public:

  enum TextAlign { alignLeft, alignCenter, alignRight };

  bool _visible = true;
  std::vector<VkCommandBuffer> _cmdBuffers;

  TextOverlay(
    vks::VulkanDevice *vulkanDevice,
    VkQueue queue,
    std::vector<VkFramebuffer> &framebuffers,
    VkFormat colorformat,
    VkFormat depthformat,
    uint32_t *framebufferwidth,
    uint32_t *framebufferheight,
    std::vector<VkPipelineShaderStageCreateInfo> shaderstages)
  {
    this->_vulkanDevice = vulkanDevice;
    this->_queue = queue;
    this->_colorFormat = colorformat;
    this->_depthFormat = depthformat;

    this->_frameBuffers.resize(framebuffers.size());
    for (uint32_t i = 0; i < framebuffers.size(); i++)
    {
      this->_frameBuffers[i] = &framebuffers[i];
    }

    this->_shaderStages = shaderstages;

    this->_frameBufferWidth = framebufferwidth;
    this->_frameBufferHeight = framebufferheight;

    _cmdBuffers.resize(framebuffers.size());
    prepareResources();
    prepareRenderPass();
    preparePipeline();
  }

  ~TextOverlay()
  {
    // Free up all Vulkan resources requested by the text overlay
    vkDestroySampler(_vulkanDevice->logicalDevice, _sampler, nullptr);
    vkDestroyImage(_vulkanDevice->logicalDevice, _image, nullptr);
    vkDestroyImageView(_vulkanDevice->logicalDevice, _view, nullptr);
    vkDestroyBuffer(_vulkanDevice->logicalDevice, _buffer, nullptr);
    vkFreeMemory(_vulkanDevice->logicalDevice, _memory, nullptr);
    vkFreeMemory(_vulkanDevice->logicalDevice, _imageMemory, nullptr);
    vkDestroyDescriptorSetLayout(_vulkanDevice->logicalDevice, _descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(_vulkanDevice->logicalDevice, _descriptorPool, nullptr);
    vkDestroyPipelineLayout(_vulkanDevice->logicalDevice, _pipelineLayout, nullptr);
    vkDestroyPipelineCache(_vulkanDevice->logicalDevice, _pipelineCache, nullptr);
    vkDestroyPipeline(_vulkanDevice->logicalDevice, _pipeline, nullptr);
    vkDestroyRenderPass(_vulkanDevice->logicalDevice, _renderPass, nullptr);
    vkDestroyCommandPool(_vulkanDevice->logicalDevice, _commandPool, nullptr);
  }

  

  // Prepare all vulkan resources required to render the font
  // The text overlay uses separate resources for descriptors (pool, sets, layouts), pipelines and command buffers
  void prepareResources()
  {
    const uint32_t fontWidth = STB_FONT_consolas_24_latin1_BITMAP_WIDTH;
    const uint32_t fontHeight = STB_FONT_consolas_24_latin1_BITMAP_WIDTH;

    static unsigned char font24pixels[fontWidth][fontHeight];
    stb_font_consolas_24_latin1(_stbFontData, font24pixels, fontHeight);

    // Command buffer

    // Pool
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = _vulkanDevice->queueFamilyIndices.graphics;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(_vulkanDevice->logicalDevice, &cmdPoolInfo, nullptr, &_commandPool));

    VkCommandBufferAllocateInfo cmdBufAllocateInfo =
      vks::initializers::commandBufferAllocateInfo(
        _commandPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        (uint32_t)_cmdBuffers.size());

    VK_CHECK_RESULT(vkAllocateCommandBuffers(_vulkanDevice->logicalDevice, &cmdBufAllocateInfo, _cmdBuffers.data()));

    // Vertex buffer
    VkDeviceSize bufferSize = TEXTOVERLAY_MAX_CHAR_COUNT * sizeof(glm::vec4);

    VkBufferCreateInfo bufferInfo = vks::initializers::bufferCreateInfo(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bufferSize);
    VK_CHECK_RESULT(vkCreateBuffer(_vulkanDevice->logicalDevice, &bufferInfo, nullptr, &_buffer));

    VkMemoryRequirements memReqs;
    VkMemoryAllocateInfo allocInfo = vks::initializers::memoryAllocateInfo();

    vkGetBufferMemoryRequirements(_vulkanDevice->logicalDevice, _buffer, &memReqs);
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = _vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VK_CHECK_RESULT(vkAllocateMemory(_vulkanDevice->logicalDevice, &allocInfo, nullptr, &_memory));
    VK_CHECK_RESULT(vkBindBufferMemory(_vulkanDevice->logicalDevice, _buffer, _memory, 0));

    // Font texture
    VkImageCreateInfo imageInfo = vks::initializers::imageCreateInfo();
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8_UNORM;
    imageInfo.extent.width = fontWidth;
    imageInfo.extent.height = fontHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK_RESULT(vkCreateImage(_vulkanDevice->logicalDevice, &imageInfo, nullptr, &_image));

    vkGetImageMemoryRequirements(_vulkanDevice->logicalDevice, _image, &memReqs);
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = _vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK_RESULT(vkAllocateMemory(_vulkanDevice->logicalDevice, &allocInfo, nullptr, &_imageMemory));
    VK_CHECK_RESULT(vkBindImageMemory(_vulkanDevice->logicalDevice, _image, _imageMemory, 0));

    // Staging

    struct {
      VkDeviceMemory memory;
      VkBuffer buffer;
    } stagingBuffer;

    VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
    bufferCreateInfo.size = allocInfo.allocationSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK_RESULT(vkCreateBuffer(_vulkanDevice->logicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer.buffer));

    // Get memory requirements for the staging buffer (alignment, memory type bits)
    vkGetBufferMemoryRequirements(_vulkanDevice->logicalDevice, stagingBuffer.buffer, &memReqs);

    allocInfo.allocationSize = memReqs.size;
    // Get memory type index for a host visible buffer
    allocInfo.memoryTypeIndex = _vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VK_CHECK_RESULT(vkAllocateMemory(_vulkanDevice->logicalDevice, &allocInfo, nullptr, &stagingBuffer.memory));
    VK_CHECK_RESULT(vkBindBufferMemory(_vulkanDevice->logicalDevice, stagingBuffer.buffer, stagingBuffer.memory, 0));

    uint8_t *data;
    VK_CHECK_RESULT(vkMapMemory(_vulkanDevice->logicalDevice, stagingBuffer.memory, 0, allocInfo.allocationSize, 0, (void **)&data));
    // Size of the font texture is WIDTH * HEIGHT * 1 byte (only one channel)
    memcpy(data, &font24pixels[0][0], fontWidth * fontHeight);
    vkUnmapMemory(_vulkanDevice->logicalDevice, stagingBuffer.memory);

    // Copy to image

    VkCommandBuffer copyCmd;
    cmdBufAllocateInfo.commandBufferCount = 1;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(_vulkanDevice->logicalDevice, &cmdBufAllocateInfo, &copyCmd));

    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cmdBufInfo));

    // Prepare for transfer
    vks::tools::setImageLayout(
      copyCmd,
      _image,
      VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkBufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = fontWidth;
    bufferCopyRegion.imageExtent.height = fontHeight;
    bufferCopyRegion.imageExtent.depth = 1;

    vkCmdCopyBufferToImage(
      copyCmd,
      stagingBuffer.buffer,
      _image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      1,
      &bufferCopyRegion
    );

    // Prepare for shader read
    vks::tools::setImageLayout(
      copyCmd,
      _image,
      VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));

    VkSubmitInfo submitInfo = vks::initializers::submitInfo();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &copyCmd;

    VK_CHECK_RESULT(vkQueueSubmit(_queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK_RESULT(vkQueueWaitIdle(_queue));

    vkFreeCommandBuffers(_vulkanDevice->logicalDevice, _commandPool, 1, &copyCmd);
    vkFreeMemory(_vulkanDevice->logicalDevice, stagingBuffer.memory, nullptr);
    vkDestroyBuffer(_vulkanDevice->logicalDevice, stagingBuffer.buffer, nullptr);

    VkImageViewCreateInfo imageViewInfo = vks::initializers::imageViewCreateInfo();
    imageViewInfo.image = _image;
    imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.format = imageInfo.format;
    imageViewInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,	VK_COMPONENT_SWIZZLE_A };
    imageViewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VK_CHECK_RESULT(vkCreateImageView(_vulkanDevice->logicalDevice, &imageViewInfo, nullptr, &_view));

    // Sampler
    VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(_vulkanDevice->logicalDevice, &samplerInfo, nullptr, &_sampler));

    // Descriptor
    // Font uses a separate descriptor pool
    std::array<VkDescriptorPoolSize, 1> poolSizes;
    poolSizes[0] = vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1);

    VkDescriptorPoolCreateInfo descriptorPoolInfo =
      vks::initializers::descriptorPoolCreateInfo(
        static_cast<uint32_t>(poolSizes.size()),
        poolSizes.data(),
        1);

    VK_CHECK_RESULT(vkCreateDescriptorPool(_vulkanDevice->logicalDevice, &descriptorPoolInfo, nullptr, &_descriptorPool));

    // Descriptor set layout
    std::array<VkDescriptorSetLayoutBinding, 1> setLayoutBindings;
    setLayoutBindings[0] = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo =
      vks::initializers::descriptorSetLayoutCreateInfo(
        setLayoutBindings.data(),
        static_cast<uint32_t>(setLayoutBindings.size()));
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(_vulkanDevice->logicalDevice, &descriptorSetLayoutInfo, nullptr, &_descriptorSetLayout));

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo =
      vks::initializers::pipelineLayoutCreateInfo(
        &_descriptorSetLayout,
        1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(_vulkanDevice->logicalDevice, &pipelineLayoutInfo, nullptr, &_pipelineLayout));

    // Descriptor set
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo =
      vks::initializers::descriptorSetAllocateInfo(
        _descriptorPool,
        &_descriptorSetLayout,
        1);

    VK_CHECK_RESULT(vkAllocateDescriptorSets(_vulkanDevice->logicalDevice, &descriptorSetAllocInfo, &_descriptorSet));

    VkDescriptorImageInfo texDescriptor =
      vks::initializers::descriptorImageInfo(
        _sampler,
        _view,
        VK_IMAGE_LAYOUT_GENERAL);

    std::array<VkWriteDescriptorSet, 1> writeDescriptorSets;
    writeDescriptorSets[0] = vks::initializers::writeDescriptorSet(_descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &texDescriptor);
    vkUpdateDescriptorSets(_vulkanDevice->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);

    // Pipeline cache
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VK_CHECK_RESULT(vkCreatePipelineCache(_vulkanDevice->logicalDevice, &pipelineCacheCreateInfo, nullptr, &_pipelineCache));
  }

  // Prepare a separate pipeline for the font rendering decoupled from the main application
  void preparePipeline()
  {
    // Enable blending, using alpha from red channel of the font texture (see text.frag)
    VkPipelineColorBlendAttachmentState blendAttachmentState{};
    blendAttachmentState.blendEnable = VK_TRUE;
    blendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE, 0);
    VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
    std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

    std::array<VkVertexInputBindingDescription, 2> vertexInputBindings = {
      vks::initializers::vertexInputBindingDescription(0, sizeof(glm::vec4), VK_VERTEX_INPUT_RATE_VERTEX),
      vks::initializers::vertexInputBindingDescription(1, sizeof(glm::vec4), VK_VERTEX_INPUT_RATE_VERTEX),
    };
    std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributes = {
      vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32_SFLOAT, 0),					// Location 0: Position
      vks::initializers::vertexInputAttributeDescription(1, 1, VK_FORMAT_R32G32_SFLOAT, sizeof(glm::vec2)),	// Location 1: UV
    };

    VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
    vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
    vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(_pipelineLayout, _renderPass, 0);
    pipelineCreateInfo.pVertexInputState = &vertexInputState;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(_shaderStages.size());
    pipelineCreateInfo.pStages = _shaderStages.data();

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(_vulkanDevice->logicalDevice, _pipelineCache, 1, &pipelineCreateInfo, nullptr, &_pipeline));
  }

  // Prepare a separate render pass for rendering the text as an overlay
  void prepareRenderPass()
  {
    VkAttachmentDescription attachments[2] = {};

    // Color attachment
    attachments[0].format = _colorFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    // Don't clear the framebuffer (like the renderpass from the example does)
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Depth attachment
    attachments[1].format = _depthFormat;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorReference = {};
    colorReference.attachment = 0;
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthReference = {};
    depthReference.attachment = 1;
    depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Use subpass dependencies for image layout transitions
    VkSubpassDependency subpassDependencies[2] = {};

    // Transition from final to initial (VK_SUBPASS_EXTERNAL refers to all commmands executed outside of the actual renderpass)
    subpassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    subpassDependencies[0].dstSubpass = 0;
    subpassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    subpassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    subpassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    subpassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Transition from initial to final
    subpassDependencies[1].srcSubpass = 0;
    subpassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    subpassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subpassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    subpassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    subpassDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    subpassDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.flags = 0;
    subpassDescription.inputAttachmentCount = 0;
    subpassDescription.pInputAttachments = NULL;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;
    subpassDescription.pResolveAttachments = NULL;
    subpassDescription.pDepthStencilAttachment = &depthReference;
    subpassDescription.preserveAttachmentCount = 0;
    subpassDescription.pPreserveAttachments = NULL;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.pNext = NULL;
    renderPassInfo.attachmentCount = 2;
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    renderPassInfo.dependencyCount = 2;
    renderPassInfo.pDependencies = subpassDependencies;

    VK_CHECK_RESULT(vkCreateRenderPass(_vulkanDevice->logicalDevice, &renderPassInfo, nullptr, &_renderPass));
  }

  // Map buffer 
  void beginTextUpdate()
  {
    VK_CHECK_RESULT(vkMapMemory(_vulkanDevice->logicalDevice, _memory, 0, VK_WHOLE_SIZE, 0, (void **)&_mapped));
    _numLetters = 0;
  }

  // Add text to the current buffer
  // todo : drop shadow? color attribute?
  void addText(std::string text, float x, float y, TextAlign align)
  {
    const uint32_t firstChar = STB_FONT_consolas_24_latin1_FIRST_CHAR;

    assert(_mapped != nullptr);

    const float charW = 1.5f / *_frameBufferWidth;
    const float charH = 1.5f / *_frameBufferHeight;

    float fbW = (float)*_frameBufferWidth;
    float fbH = (float)*_frameBufferHeight;
    x = (x / fbW * 2.0f) - 1.0f;
    y = (y / fbH * 2.0f) - 1.0f;

    // Calculate text width
    float textWidth = 0;
    for (auto letter : text)
    {
      stb_fontchar *charData = &_stbFontData[(uint32_t)letter - firstChar];
      textWidth += charData->advance * charW;
    }

    switch (align)
    {
    case alignRight:
      x -= textWidth;
      break;
    case alignCenter:
      x -= textWidth / 2.0f;
      break;
    }

    // Generate a uv mapped quad per char in the new text
    for (auto letter : text)
    {
      stb_fontchar *charData = &_stbFontData[(uint32_t)letter - firstChar];

      _mapped->x = (x + (float)charData->x0 * charW);
      _mapped->y = (y + (float)charData->y0 * charH);
      _mapped->z = charData->s0;
      _mapped->w = charData->t0;
      _mapped++;

      _mapped->x = (x + (float)charData->x1 * charW);
      _mapped->y = (y + (float)charData->y0 * charH);
      _mapped->z = charData->s1;
      _mapped->w = charData->t0;
      _mapped++;

      _mapped->x = (x + (float)charData->x0 * charW);
      _mapped->y = (y + (float)charData->y1 * charH);
      _mapped->z = charData->s0;
      _mapped->w = charData->t1;
      _mapped++;

      _mapped->x = (x + (float)charData->x1 * charW);
      _mapped->y = (y + (float)charData->y1 * charH);
      _mapped->z = charData->s1;
      _mapped->w = charData->t1;
      _mapped++;

      x += charData->advance * charW;

      _numLetters++;
    }
  }

  // Unmap buffer and update command buffers
  void endTextUpdate()
  {
    vkUnmapMemory(_vulkanDevice->logicalDevice, _memory);
    _mapped = nullptr;
    updateCommandBuffers();
  }

  // Needs to be called by the application
  void updateCommandBuffers()
  {
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = _renderPass;
    renderPassBeginInfo.renderArea.extent.width = *_frameBufferWidth;
    renderPassBeginInfo.renderArea.extent.height = *_frameBufferHeight;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

    for (int32_t i = 0; i < _cmdBuffers.size(); ++i)
    {
      renderPassBeginInfo.framebuffer = *_frameBuffers[i];

      VK_CHECK_RESULT(vkBeginCommandBuffer(_cmdBuffers[i], &cmdBufInfo));

      vkCmdBeginRenderPass(_cmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

      VkViewport viewport = vks::initializers::viewport((float)*_frameBufferWidth, (float)*_frameBufferHeight, 0.0f, 1.0f);
      vkCmdSetViewport(_cmdBuffers[i], 0, 1, &viewport);

      VkRect2D scissor = vks::initializers::rect2D(*_frameBufferWidth, *_frameBufferHeight, 0, 0);
      vkCmdSetScissor(_cmdBuffers[i], 0, 1, &scissor);

      vkCmdBindPipeline(_cmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline);
      vkCmdBindDescriptorSets(_cmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineLayout, 0, 1, &_descriptorSet, 0, NULL);

      VkDeviceSize offsets = 0;
      vkCmdBindVertexBuffers(_cmdBuffers[i], 0, 1, &_buffer, &offsets);
      vkCmdBindVertexBuffers(_cmdBuffers[i], 1, 1, &_buffer, &offsets);
      for (uint32_t j = 0; j < _numLetters; j++)
      {
        vkCmdDraw(_cmdBuffers[i], 4, 1, j * 4, 0);
      }


      vkCmdEndRenderPass(_cmdBuffers[i]);

      VK_CHECK_RESULT(vkEndCommandBuffer(_cmdBuffers[i]));
    }
  }

  // Submit the text command buffers to a queue
  // Does a queue wait idle
  void submit(VkQueue queue, uint32_t bufferindex)
  {
    if (!_visible)
    {
      return;
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &_cmdBuffers[bufferindex];

    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK_RESULT(vkQueueWaitIdle(queue));
  }

};



struct arena_vertex {
  float position[3];
  float colorRGBA[4];

  void set_color(float color[4])
  {
    colorRGBA[0] = color[0];
    colorRGBA[1] = color[1];
    colorRGBA[2] = color[2];
    colorRGBA[3] = color[3];
  }

  void transform(float rCos, float rSin, float rDisplace)
  {

    position[0] = position[0] + rDisplace;

    float ap0 = position[0] * rCos - position[1] * rSin;
    float ap1 = position[0] * rSin + position[1] * rCos;

    position[0] = ap0;
    position[1] = ap1;

    position[2] += 0.3f * sqrt(ap0*ap0 + ap1 * ap1);
    
  }

};

class VulkanExample : public VulkanExampleBase
{
public:

	// Vertex buffer and attributes
	struct {
		VkDeviceMemory memory;															// Handle to the device memory for this buffer
		VkBuffer buffer;																// Handle to the Vulkan buffer object that the memory is bound to
	} arena_vertices;

	// Index buffer
	struct 
	{
		VkDeviceMemory memory;		
		VkBuffer buffer;			
		uint32_t count;
	} arena_indices;

	// Uniform buffer block object
	struct {
		VkDeviceMemory memory;		
		VkBuffer buffer;			
		VkDescriptorBufferInfo descriptor;
	}  arena_uniformBufferVS;

	// For simplicity we use the same uniform block layout as in the shader:
	//
	//	layout(set = 0, binding = 0) uniform UBO
	//	{
	//		mat4 projectionMatrix;
	//		mat4 modelMatrix;
	//		mat4 viewMatrix;
	//	} ubo;
	//
	// This way we can just memcopy the ubo data to the ubo
	// Note: You should use data types that align with the GPU in order to avoid manual padding (vec4, mat4)
	struct {
		glm::mat4 projectionMatrix;
		glm::mat4 modelMatrix;
		glm::mat4 viewMatrix;
	} arena_uboVS;

	// The pipeline layout is used by a pipeline to access the descriptor sets 
	// It defines interface (without binding any actual data) between the shader stages used by the pipeline and the shader resources
	// A pipeline layout can be shared among multiple pipelines as long as their interfaces match
	VkPipelineLayout arena_pipelineLayout;

	// Pipelines (often called "pipeline state objects") are used to bake all states that affect a pipeline
	// While in OpenGL every state can be changed at (almost) any time, Vulkan requires to layout the graphics (and compute) pipeline states upfront
	// So for each combination of non-dynamic pipeline states you need a new pipeline (there are a few exceptions to this not discussed here)
	// Even though this adds a new dimension of planing ahead, it's a great opportunity for performance optimizations by the driver
	VkPipeline arena_pipeline;

	// The descriptor set layout describes the shader binding layout (without actually referencing descriptor)
	// Like the pipeline layout it's pretty much a blueprint and can be used with different descriptor sets as long as their layout matches
	VkDescriptorSetLayout arena_descriptorSetLayout;

	// The descriptor set stores the resources bound to the binding points in a shader
	// It connects the binding points of the different shaders with the buffers and images used for those bindings
	VkDescriptorSet arena_descriptorSet;


	// Synchronization primitives
	// Synchronization is an important concept of Vulkan that OpenGL mostly hid away. Getting this right is crucial to using Vulkan.

	// Semaphores
	// Used to coordinate operations within the graphics queue and ensure correct command ordering
	VkSemaphore presentCompleteSemaphore;
	VkSemaphore renderCompleteSemaphore;

	// Fences
	// Used to check the completion of queue operations (e.g. command buffer execution)
	std::vector<VkFence> waitFences;

  TextOverlay *textOverlay = nullptr;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{

		zoom = -125.0f;
		title = "Visualisering: Beholding lokal vs. regional. Konseptfase.";
		// Values not set here are initialized in the base class constructor
    settings.overlay = false;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note: Inherited destructor cleans up resources stored in base class
    // Clean up used Vulkan resources 
    // Note : Inherited destructor cleans up resources stored in base class

    // Clean up texture resources

		vkDestroyPipeline(device, arena_pipeline, nullptr);

		vkDestroyPipelineLayout(device, arena_pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, arena_descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, arena_vertices.buffer, nullptr);
		vkFreeMemory(device, arena_vertices.memory, nullptr);

		vkDestroyBuffer(device, arena_indices.buffer, nullptr);
		vkFreeMemory(device, arena_indices.memory, nullptr);

		vkDestroyBuffer(device, arena_uniformBufferVS.buffer, nullptr);
		vkFreeMemory(device, arena_uniformBufferVS.memory, nullptr);


    if (textOverlay != nullptr)
    {
      delete(textOverlay);
      textOverlay = nullptr;
    }
   

		vkDestroySemaphore(device, presentCompleteSemaphore, nullptr);
		vkDestroySemaphore(device, renderCompleteSemaphore, nullptr);

		for (auto& fence : waitFences)
		{
			vkDestroyFence(device, fence, nullptr);
		}
	}

  void prepareTextOverlay()
  {
    // Load the text rendering shaders
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    shaderStages.push_back(loadShader(getAssetPath() + "shaders/textoverlay/text.vert.spv", VK_SHADER_STAGE_VERTEX_BIT));
    shaderStages.push_back(loadShader(getAssetPath() + "shaders/textoverlay/text.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT));

    textOverlay = new TextOverlay(
      vulkanDevice,
      queue,
      frameBuffers,
      swapChain.colorFormat,
      depthFormat,
      &width,
      &height,
      shaderStages
    );
    updateTextOverlay();
  }
 

  // Update the text buffer displayed by the text overlay
  void updateTextOverlay(void)
  {
    textOverlay->beginTextUpdate();

    textOverlay->addText(title, 5.0f, 5.0f, TextOverlay::alignLeft);

    float arena_rotationX = arena_uboVS.modelMatrix[0][0];
    float arena_rotationY = arena_uboVS.modelMatrix[1][0];

    float rAngle = std::atan2(arena_rotationY, arena_rotationX);

    if (rAngle < 0.f)
    {
      rAngle += float(2.0 * E_PI);
    }

    rAngle /= float(2.0 * E_PI);   // [0..1]

    rAngle -= 0.75f;   // Some magic to fit matrix values. Clean up.

    if (rAngle < 0.f)
    {
      rAngle += 1.f;
    }

    rAngle *= float(2.0 * E_PI);

    float fRay = rAngle * 2.0f / float(E_PI) / 0.05f;

    int nRay = int(fRay + .5f);

    textOverlay->addText(title, 5.0f, 5.0f, TextOverlay::alignLeft);

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << (frameTimer * 1000.0f) << "ms (" << lastFPS << " fps) " << deviceProperties.deviceName << " rotation " << rAngle << " nRay " << nRay;
    textOverlay->addText(ss.str(), 5.0f, 25.0f, TextOverlay::alignLeft);

    // Display current model view matrix
    textOverlay->addText("model view matrix", (float)width, 5.0f, TextOverlay::alignRight);


    for (uint32_t i = 0; i < 4; i++)
    {
      ss.str("");
      ss << std::fixed << std::setprecision(2) << std::showpos;
      ss << arena_uboVS.modelMatrix[0][i] << " " << arena_uboVS.modelMatrix[1][i] << " " << arena_uboVS.modelMatrix[2][i] << " " << arena_uboVS.modelMatrix[3][i];
      textOverlay->addText(ss.str(), (float)width, 25.0f + (float)i * 20.0f, TextOverlay::alignRight);
    }

    glm::vec3 projected = glm::project(glm::vec3(34.6281052f, 20.0473022f, 11.0037498f), arena_uboVS.modelMatrix, arena_uboVS.projectionMatrix, glm::vec4(0, 0, (float)width, (float)height));

    // textOverlay->addText("IRAY7.I5", projected.x, projected.y, TextOverlay::alignCenter);
    textOverlay->addText("[X]", projected.x, projected.y, TextOverlay::alignCenter);

    textOverlay->addText("Syntetiske data.", 5.0f, 65.0f, TextOverlay::alignLeft);
    textOverlay->addText("Prosjektsamarbeid med NAV Alna. NAV IT Data og Innsikt. AI-lab. Sept 2018.", 5.0f, 85.0f, TextOverlay::alignLeft);

    textOverlay->endTextUpdate();
  }

	// This function is used to request a device memory type that supports all the property flags we request (e.g. device local, host visibile)
	// Upon success it will return the index of the memory type that fits our requestes memory properties
	// This is necessary as implementations can offer an arbitrary number of memory types with different
	// memory properties. 
	// You can check http://vulkan.gpuinfo.org/ for details on different memory configurations
	uint32_t getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties)
	{
		// Iterate over all memory types available for the device used in this example
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
		{
			if ((typeBits & 1) == 1)
			{
				if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{						
					return i;
				}
			}
			typeBits >>= 1;
		}

		throw "Could not find a suitable memory type!";
	}

	// Create the Vulkan synchronization primitives used in this example
	void prepareSynchronizationPrimitives()
	{
		// Semaphores (Used for correct command ordering)
		VkSemaphoreCreateInfo semaphoreCreateInfo = {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		semaphoreCreateInfo.pNext = nullptr;

		// Semaphore used to ensures that image presentation is complete before starting to submit again
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &presentCompleteSemaphore));

		// Semaphore used to ensures that all commands submitted have been finished before submitting the image to the queue
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderCompleteSemaphore));

		// Fences (Used to check draw command buffer completion)
		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		// Create in signaled state so we don't wait on first render of each command buffer
		fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		waitFences.resize(drawCmdBuffers.size());
		for (auto& fence : waitFences)
		{
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
		}
	}

	// Get a new command buffer from the command pool
	// If begin is true, the command buffer is also started so we can start adding commands
	VkCommandBuffer getCommandBuffer(bool begin)
	{
		VkCommandBuffer cmdBuffer;

		VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
		cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocateInfo.commandPool = cmdPool;
		cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocateInfo.commandBufferCount = 1;
	
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &cmdBuffer));

		// If requested, also start the new command buffer
		if (begin)
		{
			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
			VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
		}

		return cmdBuffer;
	}

	// End the command buffer and submit it to the queue
	// Uses a fence to ensure command buffer has finished executing before deleting it
	void flushCommandBuffer(VkCommandBuffer commandBuffer)
	{
		assert(commandBuffer != VK_NULL_HANDLE);

		VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		// Create fence to ensure that the command buffer has finished executing
		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = 0;
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

		// Submit to the queue
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));

		// Wait for the fence to signal that command buffer has finished executing
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));

		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, cmdPool, 1, &commandBuffer);
	}

	// Build separate command buffers for every framebuffer image
	// Unlike in OpenGL all rendering commands are recorded once into command buffers that are then resubmitted to the queue
	// This allows to generate work upfront and from multiple threads, one of the biggest advantages of Vulkan
	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = {};
		cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmdBufInfo.pNext = nullptr;

		// Set clear values for all framebuffer attachments with loadOp set to clear
		// We use two attachments (color and depth) that are cleared at the start of the subpass and as such we need to set clear values for both
		VkClearValue clearValues[2];
		clearValues[0].color = { { 0.1f, 0.1f, 0.3f, 1.0f } };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = {};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.pNext = nullptr;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;
	
		for (uint32_t iCmdBuffer = 0; iCmdBuffer < drawCmdBuffers.size(); ++iCmdBuffer)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[iCmdBuffer];

      const VkCommandBuffer& cmdBuffer = drawCmdBuffers[iCmdBuffer];


			VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

			// Start the first sub pass specified in our default render pass setup by the base class
			// This will clear the color and depth attachment
			vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			// Update dynamic viewport state
			VkViewport viewport = {};
			viewport.height = (float)height;
			viewport.width = (float)width;
			viewport.minDepth = (float) 0.0f;
			viewport.maxDepth = (float) 1.0f;
			vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

			// Update dynamic scissor state
			VkRect2D scissor = {};
			scissor.extent.width = width;
			scissor.extent.height = height;
			scissor.offset.x = 0;
			scissor.offset.y = 0;
			vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

			// Bind descriptor sets describing shader binding points

			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, arena_pipelineLayout, 0, 1, &arena_descriptorSet, 0, nullptr);


			// Bind the rendering pipeline
			// The pipeline (state object) contains all states of the rendering pipeline, binding it will set all the states specified at pipeline creation time
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, arena_pipeline);

			// Bind triangle vertex buffer (contains position and colors)
			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &arena_vertices.buffer, offsets);

			// Bind triangle index buffer
			vkCmdBindIndexBuffer(cmdBuffer, arena_indices.buffer, 0, VK_INDEX_TYPE_UINT32);

			// Draw indexed triangle
			vkCmdDrawIndexed(cmdBuffer, arena_indices.count, 1, 0, 0, 1);

			vkCmdEndRenderPass(cmdBuffer);

			// Ending the render pass will add an implicit barrier transitioning the frame buffer color attachment to 
			// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR for presenting it to the windowing system

			VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
		}
	}

	void draw()
	{
		// Get next image in the swap chain (back/front buffer)
		VK_CHECK_RESULT(swapChain.acquireNextImage(presentCompleteSemaphore, &currentBuffer));

		// Use a fence to wait until the command buffer has finished execution before using it again
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX));
		VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentBuffer]));

		// Pipeline stage at which the queue submission will wait (via pWaitSemaphores)
		VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		// The submit info structure specifices a command buffer queue submission batch
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pWaitDstStageMask = &waitStageMask;									// Pointer to the list of pipeline stages that the semaphore waits will occur at
		submitInfo.pWaitSemaphores = &presentCompleteSemaphore;							// Semaphore(s) to wait upon before the submitted command buffer starts executing
		submitInfo.waitSemaphoreCount = 1;												// One wait semaphore																				
		submitInfo.pSignalSemaphores = &renderCompleteSemaphore;						// Semaphore(s) to be signaled when command buffers have completed
		submitInfo.signalSemaphoreCount = 1;											// One signal semaphore
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];					// Command buffers(s) to execute in this batch (submission)
		submitInfo.commandBufferCount = 1;												// One command buffer

		


    if (textOverlay->_visible)
    {
      VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

      VkSubmitInfo submitInfo_text = {};
      submitInfo_text.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo_text.commandBufferCount = 1;
      submitInfo_text.pCommandBuffers = &textOverlay->_cmdBuffers[currentBuffer];

      VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo_text, waitFences[currentBuffer]));
      VK_CHECK_RESULT(vkQueueWaitIdle(queue));

    }
    else
    {
      // Submit to the graphics queue passing a wait fence
      VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentBuffer]));
    }


		
		// Present the current buffer to the swap chain
		// Pass the semaphore signaled by the command buffer submission from the submit info as the wait semaphore for swap chain presentation
		// This ensures that the image is not presented to the windowing system until all commands have been submitted
		VK_CHECK_RESULT(swapChain.queuePresent(queue, currentBuffer, renderCompleteSemaphore));
	}

  

  

  
	// Prepare vertex and index buffers for an indexed triangle
	// Also uploads them to device local memory using staging and initializes vertex input and attribute binding to match the vertex shader
  void arena_prepareVertices()
  {
    // A note on memory management in Vulkan in general:
    // This is a very complex topic and while it's fine for an example application to to small individual memory allocations that is not
    // what should be done a real-world application, where you should allocate large chunkgs of memory at once isntead.

    // Setup vertices

    std::vector<arena_vertex> lcVertex;

    const float x_0 = -1.0f;
    const float y_0 = -1.f;
    const float z_0 = -1.f;
  

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(1, 6);

    std::uniform_int_distribution<> color_dis(0, 6);
    std::uniform_int_distribution<> distance_dis(1, 10);

    std::uniform_int_distribution<> z_l_dis(0, 5);

    std::uniform_int_distribution<> y_l_dis(0, 5);

    std::uniform_int_distribution<> skip_dis(0, 10);



    for (uint32_t iRay = 0; iRay < 71; iRay++)
    {

      if (iRay > 50 && iRay < 60)
      {
        continue;
      }

      const float PI = 3.1415927f;

      const float rAngle = 0.05f* iRay * PI / 2.f;

      const float rCos = cos(rAngle);
      const float rSin = sin(rAngle);

      float x_init = 15.f;
      const float x_offset_step = 7.0f;

      for (uint32_t i = 0; i < 10; i++)
      {
        const float y_L = 2.0f - y_l_dis(gen) * 0.2f;
        const float y_1 = y_0 + y_L;

        const float z_L = 0.7f + z_l_dis(gen) * 0.1f;
        const float z_1 = z_0 + z_L;

        const float x_L = 2.0f + dis(gen) * 0.3f;
        const float x_1 = x_0 + x_L;


        struct Cube {

          arena_vertex acVertex[8];

          Cube(float x0, float y0, float z0, float x1, float y1, float z1)
          {
            acVertex[0] = { { x0, y0, z0 },{ 0,0,0,0 } };
            acVertex[1] = { { x1, y0, z0 },{ 0,0,0,0 } };
            acVertex[2]=  { { x1,  y1, z0 },{ 0,0,0,0 } };
            acVertex[3] = { { x0,  y1, z0 },{ 0,0,0,0 } };

            acVertex[4] = { { x0, y0,  z1 },{ 0,0,0,0 } };
            acVertex[5] = { { x1, y0,  z1 },{ 0,0,0,0 } };
            acVertex[6] = { { x1,  y1,  z1 },{ 0,0,0,0 } };
            acVertex[7] = { { x0,  y1,  z1 },{ 0,0,0,0 } };

          }

          void transform(float rCos, float rSin, float x)
          {
            for (int iVertex = 0; iVertex < 8; iVertex++)
            {
              acVertex[iVertex].transform(rCos, rSin, x);
            }

          }

          void set_color_low_x(float color[4])
          {
            acVertex[0].set_color(color);
            acVertex[3].set_color(color);
            acVertex[4].set_color(color);
            acVertex[7].set_color(color);
          }

          void set_color_hi_x(float color[4])
          {
            acVertex[1].set_color(color);
            acVertex[2].set_color(color);
            acVertex[5].set_color(color);
            acVertex[6].set_color(color);
          }

          void add(std::vector<arena_vertex>& lcVertex)
          {
            for (int iVertex = 0; iVertex < 8; iVertex++)
            {
              lcVertex.push_back(acVertex[iVertex]);
            }
          }

        };

       
       
       

        const float x_offset_this = x_offset_step  + 0.4f * distance_dis(gen);

        float x = x_init + x_offset_this;

        x_init = x;

        if (skip_dis(gen) > 5)
        {
          continue;
        }


        Cube cc(x_0, y_0, z_0, x_1, y_1, z_1);

        cc.transform(rCos, rSin, x);

        int iColor = color_dis(gen);


        int nColor[3];

        if (iColor == 0)
        {
          nColor[0] = 255;
          nColor[1] = 0;
          nColor[2] = 0;
        }
        else if (iColor == 1)
        {
          nColor[0] = 255;
          nColor[1] = 127;
          nColor[2] = 0;
        }
        else if (iColor == 2)
        {
          nColor[0] = 255;
          nColor[1] = 255;
          nColor[2] = 0;
        }
        else if (iColor == 3)
        {
          nColor[0] = 0;
          nColor[1] = 255;
          nColor[2] = 0;
        }
        else if (iColor == 4)
        {
          nColor[0] = 0;
          nColor[1] = 0;
          nColor[2] = 255;
        }
        else if (iColor == 5)
        {
          nColor[0] = 75;
          nColor[1] = 0;
          nColor[2] = 130;
        }
        else if (iColor == 6)
        {
          nColor[0] = 143;
          nColor[1] = 0;
          nColor[2] = 255;
        }
        else
        {
          nColor[0] = 128;
          nColor[1] = 128;
          nColor[2] = 128;
        }

        float fColor[3];

        fColor[0] = nColor[0] / 255.f;
        fColor[1] = nColor[1] / 255.f;
        fColor[2] = nColor[2] / 255.f;

        float colorNormal[4] = { fColor[0], fColor[1], fColor[2], 1.0f };

        cc.set_color_low_x(colorNormal);
        cc.set_color_hi_x(colorNormal);

        cc.add(lcVertex);
      }
    }

		uint32_t vertexBufferSize = static_cast<uint32_t>(lcVertex.size()) * sizeof(arena_vertex);

    uint32_t num_cubes = uint32_t(lcVertex.size() / 8);

		// Setup indices

    std::vector<uint32_t> lcIndex;

    uint32_t iFirstIndex = 0;

    for (uint32_t i = 0; i < num_cubes; i++)
    {
      lcIndex.push_back(iFirstIndex + 0);
      lcIndex.push_back(iFirstIndex + 2);
      lcIndex.push_back(iFirstIndex + 1);

      lcIndex.push_back(iFirstIndex + 0);
      lcIndex.push_back(iFirstIndex + 3);
      lcIndex.push_back(iFirstIndex + 2);

      lcIndex.push_back(iFirstIndex + 1);
      lcIndex.push_back(iFirstIndex + 2);
      lcIndex.push_back(iFirstIndex + 6);

      lcIndex.push_back(iFirstIndex + 6);
      lcIndex.push_back(iFirstIndex + 5);
      lcIndex.push_back(iFirstIndex + 1);

      lcIndex.push_back(iFirstIndex + 4);
      lcIndex.push_back(iFirstIndex + 5);
      lcIndex.push_back(iFirstIndex + 6);

      lcIndex.push_back(iFirstIndex + 6);
      lcIndex.push_back(iFirstIndex + 7);
      lcIndex.push_back(iFirstIndex + 4);

      lcIndex.push_back(iFirstIndex + 2);
      lcIndex.push_back(iFirstIndex + 3);
      lcIndex.push_back(iFirstIndex + 6);

      lcIndex.push_back(iFirstIndex + 6);
      lcIndex.push_back(iFirstIndex + 3);
      lcIndex.push_back(iFirstIndex + 7);

      lcIndex.push_back(iFirstIndex + 0);
      lcIndex.push_back(iFirstIndex + 7);
      lcIndex.push_back(iFirstIndex + 3);

      lcIndex.push_back(iFirstIndex + 0);
      lcIndex.push_back(iFirstIndex + 4);
      lcIndex.push_back(iFirstIndex + 7);

      lcIndex.push_back(iFirstIndex + 0);
      lcIndex.push_back(iFirstIndex + 1);
      lcIndex.push_back(iFirstIndex + 5);

      lcIndex.push_back(iFirstIndex + 0);
      lcIndex.push_back(iFirstIndex + 5);
      lcIndex.push_back(iFirstIndex + 4);
     
      iFirstIndex += 8;
    }



	  arena_indices.count = static_cast<uint32_t>(lcIndex.size());
	  uint32_t indexBufferSize = arena_indices.count * sizeof(uint32_t);

	  VkMemoryAllocateInfo memAlloc = {};
	  memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	  VkMemoryRequirements memReqs;

	  void *data;

		
		// Static data like vertex and index buffer should be stored on the device memory 
		// for optimal (and fastest) access by the GPU
		//
		// To achieve this we use so-called "staging buffers" :
		// - Create a buffer that's visible to the host (and can be mapped)
		// - Copy the data to this buffer
		// - Create another buffer that's local on the device (VRAM) with the same size
		// - Copy the data from the host to the device using a command buffer
		// - Delete the host visible (staging) buffer
		// - Use the device local buffers for rendering

		struct StagingBuffer {
			VkDeviceMemory memory;
			VkBuffer buffer;
		};

		struct {
			StagingBuffer vertices;
			StagingBuffer indices;
		} stagingBuffers;

		// Vertex buffer
		VkBufferCreateInfo vertexBufferInfo = {};
		vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		vertexBufferInfo.size = vertexBufferSize;
		// Buffer is used as the copy source
		vertexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		// Create a host-visible buffer to copy the vertex data to (staging buffer)
		VK_CHECK_RESULT(vkCreateBuffer(device, &vertexBufferInfo, nullptr, &stagingBuffers.vertices.buffer));
		vkGetBufferMemoryRequirements(device, stagingBuffers.vertices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		// Request a host visible memory type that can be used to copy our data do
		// Also request it to be coherent, so that writes are visible to the GPU right after unmapping the buffer
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &stagingBuffers.vertices.memory));
		// Map and copy
		VK_CHECK_RESULT(vkMapMemory(device, stagingBuffers.vertices.memory, 0, memAlloc.allocationSize, 0, &data));
		memcpy(data, lcVertex.data(), vertexBufferSize);
		vkUnmapMemory(device, stagingBuffers.vertices.memory);
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffers.vertices.buffer, stagingBuffers.vertices.memory, 0));

		// Create a device local buffer to which the (host local) vertex data will be copied and which will be used for rendering
		vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VK_CHECK_RESULT(vkCreateBuffer(device, &vertexBufferInfo, nullptr, &arena_vertices.buffer));
		vkGetBufferMemoryRequirements(device, arena_vertices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &arena_vertices.memory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, arena_vertices.buffer, arena_vertices.memory, 0));

		// Index buffer
		VkBufferCreateInfo indexbufferInfo = {};
		indexbufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		indexbufferInfo.size = indexBufferSize;
		indexbufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		// Copy index data to a buffer visible to the host (staging buffer)
		VK_CHECK_RESULT(vkCreateBuffer(device, &indexbufferInfo, nullptr, &stagingBuffers.indices.buffer));
		vkGetBufferMemoryRequirements(device, stagingBuffers.indices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &stagingBuffers.indices.memory));
		VK_CHECK_RESULT(vkMapMemory(device, stagingBuffers.indices.memory, 0, indexBufferSize, 0, &data));
		memcpy(data, lcIndex.data(), indexBufferSize);
		vkUnmapMemory(device, stagingBuffers.indices.memory);
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffers.indices.buffer, stagingBuffers.indices.memory, 0));

		// Create destination buffer with device only visibility
		indexbufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VK_CHECK_RESULT(vkCreateBuffer(device, &indexbufferInfo, nullptr, &arena_indices.buffer));
		vkGetBufferMemoryRequirements(device, arena_indices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &arena_indices.memory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, arena_indices.buffer, arena_indices.memory, 0));

		// Buffer copies have to be submitted to a queue, so we need a command buffer for them
		// Note: Some devices offer a dedicated transfer queue (with only the transfer bit set) that may be faster when doing lots of copies
		VkCommandBuffer copyCmd = getCommandBuffer(true);

		// Put buffer region copies into command buffer
		VkBufferCopy copyRegion = {};

		// Vertex buffer
		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffers.vertices.buffer, arena_vertices.buffer, 1, &copyRegion);
		// Index buffer
		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffers.indices.buffer, arena_indices.buffer,	1, &copyRegion);

		// Flushing the command buffer will also submit it to the queue and uses a fence to ensure that all commands have been executed before returning
		flushCommandBuffer(copyCmd);

		// Destroy staging buffers
		// Note: Staging buffer must not be deleted before the copies have been submitted and executed
		vkDestroyBuffer(device, stagingBuffers.vertices.buffer, nullptr);
		vkFreeMemory(device, stagingBuffers.vertices.memory, nullptr);
		vkDestroyBuffer(device, stagingBuffers.indices.buffer, nullptr);
		vkFreeMemory(device, stagingBuffers.indices.memory, nullptr);
		
		
	}

	void setupDescriptorPool()
	{
		// We need to tell the API the number of max. requested descriptors per type
		VkDescriptorPoolSize typeCounts[1];
		// This example only uses one descriptor type (uniform buffer) and only requests one descriptor of this type
		typeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		typeCounts[0].descriptorCount = 1;
		// For additional types you need to add new entries in the type count list
		// E.g. for two combined image samplers :
		// typeCounts[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		// typeCounts[1].descriptorCount = 2;

		// Create the global descriptor pool
		// All descriptors used in this example are allocated from this pool
		VkDescriptorPoolCreateInfo descriptorPoolInfo = {};
		descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolInfo.pNext = nullptr;
		descriptorPoolInfo.poolSizeCount = 1;
		descriptorPoolInfo.pPoolSizes = typeCounts;
		// Set the max. number of descriptor sets that can be requested from this pool (requesting beyond this limit will result in an error)
		descriptorPoolInfo.maxSets = 1;

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void arena_setupDescriptorSetLayout()
	{
		// Setup layout of descriptors used in this example
		// Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
		// So every shader binding should map to one descriptor set layout binding

		// Binding 0: Uniform buffer (Vertex shader)
		VkDescriptorSetLayoutBinding layoutBinding = {};
		layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBinding.descriptorCount = 1;
		layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		layoutBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo descriptorLayout = {};
		descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorLayout.pNext = nullptr;
		descriptorLayout.bindingCount = 1;
		descriptorLayout.pBindings = &layoutBinding;

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &arena_descriptorSetLayout));

		// Create the pipeline layout that is used to generate the rendering pipelines that are based on this descriptor set layout
		// In a more complex scenario you would have different pipeline layouts for different descriptor set layouts that could be reused
		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
		pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pPipelineLayoutCreateInfo.pNext = nullptr;
		pPipelineLayoutCreateInfo.setLayoutCount = 1;
		pPipelineLayoutCreateInfo.pSetLayouts = &arena_descriptorSetLayout;

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &arena_pipelineLayout));
	}

	void arena_setupDescriptorSet()
	{
		// Allocate a new descriptor set from the global descriptor pool
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &arena_descriptorSetLayout;

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &arena_descriptorSet));

		// Update the descriptor set determining the shader binding points
		// For every binding point used in a shader there needs to be one
		// descriptor set matching that binding point

		VkWriteDescriptorSet writeDescriptorSet = {};

		// Binding 0 : Uniform buffer
		writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSet.dstSet = arena_descriptorSet;
		writeDescriptorSet.descriptorCount = 1;
		writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writeDescriptorSet.pBufferInfo = &arena_uniformBufferVS.descriptor;
		// Binds this uniform buffer to binding point 0
		writeDescriptorSet.dstBinding = 0;

		vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
	}

	// Create the depth (and stencil) buffer attachments used by our framebuffers
	// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
	void setupDepthStencil()
	{
		// Create an optimal image used as the depth stencil attachment
		VkImageCreateInfo image = {};
		image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = depthFormat;
		// Use example's height and width
		image.extent = { width, height, 1 };
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &depthStencil.image));

		// Allocate memory for the image (device local) and bind it to our image
		VkMemoryAllocateInfo memAlloc = {};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &depthStencil.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0));

		// Create a view for the depth stencil image
		// Images aren't directly accessed in Vulkan, but rather through views described by a subresource range
		// This allows for multiple views of one image with differing ranges (e.g. for different layers)
		VkImageViewCreateInfo depthStencilView = {};
		depthStencilView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthStencilView.format = depthFormat;
		depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;
		depthStencilView.image = depthStencil.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &depthStencil.view));
	}

	// Create a frame buffer for each swap chain image
	// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
	void setupFrameBuffer()
	{
		// Create a frame buffer for every image in the swapchain
		frameBuffers.resize(swapChain.imageCount);
		for (size_t i = 0; i < frameBuffers.size(); i++)
		{
			std::array<VkImageView, 2> attachments;										
			attachments[0] = swapChain.buffers[i].view;									// Color attachment is the view of the swapchain image			
			attachments[1] = depthStencil.view;											// Depth/Stencil attachment is the same for all frame buffers			

			VkFramebufferCreateInfo frameBufferCreateInfo = {};
			frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			// All frame buffers use the same renderpass setup
			frameBufferCreateInfo.renderPass = renderPass;
			frameBufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			frameBufferCreateInfo.pAttachments = attachments.data();
			frameBufferCreateInfo.width = width;
			frameBufferCreateInfo.height = height;
			frameBufferCreateInfo.layers = 1;
			// Create the framebuffer
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &frameBuffers[i]));
		}
	}

	// Render pass setup
	// Render passes are a new concept in Vulkan. They describe the attachments used during rendering and may contain multiple subpasses with attachment dependencies 
	// This allows the driver to know up-front what the rendering will look like and is a good opportunity to optimize especially on tile-based renderers (with multiple subpasses)
	// Using sub pass dependencies also adds implicit layout transitions for the attachment used, so we don't need to add explicit image memory barriers to transform them
	// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
	void setupRenderPass()
	{
		// This example will use a single render pass with one subpass

		// Descriptors for the attachments used by this renderpass
		std::array<VkAttachmentDescription, 2> attachments = {};

		// Color attachment
		attachments[0].format = swapChain.colorFormat;									// Use the color format selected by the swapchain
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;									// We don't use multi sampling in this example
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;							// Clear this attachment at the start of the render pass
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;							// Keep it's contents after the render pass is finished (for displaying it)
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;					// We don't use stencil, so don't care for load
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;				// Same for store
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;						// Layout at render pass start. Initial doesn't matter, so we use undefined
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;					// Layout to which the attachment is transitioned when the render pass is finished
																						// As we want to present the color buffer to the swapchain, we transition to PRESENT_KHR	
		// Depth attachment
		attachments[1].format = depthFormat;											// A proper depth format is selected in the example base
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;						
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;							// Clear depth at start of first subpass
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;						// We don't need depth after render pass has finished (DONT_CARE may result in better performance)
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;					// No stencil
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;				// No Stencil
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;						// Layout at render pass start. Initial doesn't matter, so we use undefined
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;	// Transition to depth/stencil attachment

		// Setup attachment references
		VkAttachmentReference colorReference = {};
		colorReference.attachment = 0;													// Attachment 0 is color
		colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;				// Attachment layout used as color during the subpass

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 1;													// Attachment 1 is color
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;		// Attachment used as depth/stemcil used during the subpass

		// Setup a single subpass reference
		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;			
		subpassDescription.colorAttachmentCount = 1;									// Subpass uses one color attachment
		subpassDescription.pColorAttachments = &colorReference;							// Reference to the color attachment in slot 0
		subpassDescription.pDepthStencilAttachment = &depthReference;					// Reference to the depth attachment in slot 1
		subpassDescription.inputAttachmentCount = 0;									// Input attachments can be used to sample from contents of a previous subpass
		subpassDescription.pInputAttachments = nullptr;									// (Input attachments not used by this example)
		subpassDescription.preserveAttachmentCount = 0;									// Preserved attachments can be used to loop (and preserve) attachments through subpasses
		subpassDescription.pPreserveAttachments = nullptr;								// (Preserve attachments not used by this example)
		subpassDescription.pResolveAttachments = nullptr;								// Resolve attachments are resolved at the end of a sub pass and can be used for e.g. multi sampling

		// Setup subpass dependencies
		// These will add the implicit ttachment layout transitionss specified by the attachment descriptions
		// The actual usage layout is preserved through the layout specified in the attachment reference		
		// Each subpass dependency will introduce a memory and execution dependency between the source and dest subpass described by
		// srcStageMask, dstStageMask, srcAccessMask, dstAccessMask (and dependencyFlags is set)
		// Note: VK_SUBPASS_EXTERNAL is a special constant that refers to all commands executed outside of the actual renderpass)
		std::array<VkSubpassDependency, 2> dependencies;

		// First dependency at the start of the renderpass
		// Does the transition from final to initial layout 
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;								// Producer of the dependency 
		dependencies[0].dstSubpass = 0;													// Consumer is our single subpass that will wait for the execution depdendency
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;			
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Second dependency at the end the renderpass
		// Does the transition from the initial to the final layout
		dependencies[1].srcSubpass = 0;													// Producer of the dependency is our single subpass
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;								// Consumer are all commands outside of the renderpass
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());		// Number of attachments used by this render pass
		renderPassInfo.pAttachments = attachments.data();								// Descriptions of the attachments used by the render pass
		renderPassInfo.subpassCount = 1;												// We only use one subpass in this example
		renderPassInfo.pSubpasses = &subpassDescription;								// Description of that subpass
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());	// Number of subpass dependencies
		renderPassInfo.pDependencies = dependencies.data();								// Subpass dependencies used by the render pass

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
	}

	// Vulkan loads it's shaders from an immediate binary representation called SPIR-V
	// Shaders are compiled offline from e.g. GLSL using the reference glslang compiler
	// This function loads such a shader from a binary file and returns a shader module structure
	VkShaderModule loadSPIRVShader(std::string filename)
	{
		size_t shaderSize;
		char* shaderCode = NULL;

#if defined(__ANDROID__)
		// Load shader from compressed asset
		AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, filename.c_str(), AASSET_MODE_STREAMING);
		assert(asset);
		shaderSize = AAsset_getLength(asset);
		assert(shaderSize > 0);

		shaderCode = new char[shaderSize];
		AAsset_read(asset, shaderCode, shaderSize);
		AAsset_close(asset);
#else
		std::ifstream is(filename, std::ios::binary | std::ios::in | std::ios::ate);

		if (is.is_open())
		{
			shaderSize = is.tellg();
			is.seekg(0, std::ios::beg);
			// Copy file contents into a buffer
			shaderCode = new char[shaderSize];
			is.read(shaderCode, shaderSize);
			is.close();
			assert(shaderSize > 0);
		}
#endif
		if (shaderCode)
		{
			// Create a new shader module that will be used for pipeline creation
			VkShaderModuleCreateInfo moduleCreateInfo{};
			moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			moduleCreateInfo.codeSize = shaderSize;
			moduleCreateInfo.pCode = (uint32_t*)shaderCode;

			VkShaderModule shaderModule;
			VK_CHECK_RESULT(vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule));

			delete[] shaderCode;

			return shaderModule;
		}
		else
		{
			std::cerr << "Error: Could not open shader file \"" << filename << "\"" << std::endl;
			return VK_NULL_HANDLE;
		}
	}

	void arena_preparePipelines()
	{
		// Create the graphics pipeline used in this example
		// Vulkan uses the concept of rendering pipelines to encapsulate fixed states, replacing OpenGL's complex state machine
		// A pipeline is then stored and hashed on the GPU making pipeline changes very fast
		// Note: There are still a few dynamic states that are not directly part of the pipeline (but the info that they are used is)

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
		pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		// The layout used for this pipeline (can be shared among multiple pipelines using the same layout)
		pipelineCreateInfo.layout = arena_pipelineLayout;
		// Renderpass this pipeline is attached to
		pipelineCreateInfo.renderPass = renderPass;

		// Construct the differnent states making up the pipeline

		// Input assembly state describes how primitives are assembled
		// This pipeline will assemble vertex data as a triangle lists (though we only use one triangle)
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = {};
		inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		// Rasterization state
		VkPipelineRasterizationStateCreateInfo rasterizationState = {};
		rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationState.cullMode = VK_CULL_MODE_NONE;
		rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizationState.depthClampEnable = VK_FALSE;
		rasterizationState.rasterizerDiscardEnable = VK_FALSE;
		rasterizationState.depthBiasEnable = VK_FALSE;
		rasterizationState.lineWidth = 1.0f;

		// Color blend state describes how blend factors are calculated (if used)
		// We need one blend attachment state per color attachment (even if blending is not used
		VkPipelineColorBlendAttachmentState colorBlendAttachment= {};
    
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		    
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

                                            // Viewport state sets the number of viewports and scissor used in this pipeline
		// Note: This is actually overriden by the dynamic states (see below)
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		// Enable dynamic states
		// Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
		// To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
		// For this example we will set the viewport and scissor using dynamic states
		std::vector<VkDynamicState> dynamicStateEnables;
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
		VkPipelineDynamicStateCreateInfo dynamicState = {};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.pDynamicStates = dynamicStateEnables.data();
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

		// Depth and stencil state containing depth and stencil compare and test operations
		// We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
		VkPipelineDepthStencilStateCreateInfo depthStencilState = {};
		depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencilState.depthTestEnable = VK_TRUE;
		depthStencilState.depthWriteEnable = VK_TRUE;
		depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencilState.depthBoundsTestEnable = VK_FALSE;
		depthStencilState.back.failOp = VK_STENCIL_OP_KEEP;
		depthStencilState.back.passOp = VK_STENCIL_OP_KEEP;
		depthStencilState.back.compareOp = VK_COMPARE_OP_ALWAYS;
		depthStencilState.stencilTestEnable = VK_FALSE;
		depthStencilState.front = depthStencilState.back;

		// Multi sampling state
		// This example does not make use fo multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
		VkPipelineMultisampleStateCreateInfo multisampleState = {};
		multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampleState.pSampleMask = nullptr;

		// Vertex input descriptions 
		// Specifies the vertex input parameters for a pipeline

		// Vertex input binding
		// This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)
		VkVertexInputBindingDescription vertexInputBinding = {};
		vertexInputBinding.binding = 0;
		vertexInputBinding.stride = sizeof(arena_vertex);
		vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		// Inpute attribute bindings describe shader attribute locations and memory layouts
		std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributs;
		// These match the following shader layout (see triangle.vert):
		//	layout (location = 0) in vec3 inPos;
		//	layout (location = 1) in vec3 inColor;
		// Attribute location 0: Position
		vertexInputAttributs[0].binding = 0;
		vertexInputAttributs[0].location = 0;
		// Position attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
		vertexInputAttributs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexInputAttributs[0].offset = offsetof(arena_vertex, position);
		// Attribute location 1: Color
		vertexInputAttributs[1].binding = 0;
		vertexInputAttributs[1].location = 1;
		
    // Color attribute is four 32 bit signed (SFLOAT) floats (R32 G32 B32 A32)
		vertexInputAttributs[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		vertexInputAttributs[1].offset = offsetof(arena_vertex, colorRGBA);

		// Vertex input state used for pipeline creation
		VkPipelineVertexInputStateCreateInfo vertexInputState = {};
		vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputState.vertexBindingDescriptionCount = 1;
		vertexInputState.pVertexBindingDescriptions = &vertexInputBinding;
		vertexInputState.vertexAttributeDescriptionCount = 2;
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributs.data();

		// Shaders
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		// Vertex shader
		shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// Set pipeline stage for this shader
		shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		// Load binary SPIR-V shader
		shaderStages[0].module = loadSPIRVShader(getAssetPath() + "shaders/triangle/vert.spv");
		// Main entry point for the shader
		shaderStages[0].pName = "main";
		assert(shaderStages[0].module != VK_NULL_HANDLE);

		// Fragment shader
		shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// Set pipeline stage for this shader
		shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		// Load binary SPIR-V shader
		shaderStages[1].module = loadSPIRVShader(getAssetPath() + "shaders/triangle/frag.spv");
		// Main entry point for the shader
		shaderStages[1].pName = "main";
		assert(shaderStages[1].module != VK_NULL_HANDLE);

		// Set pipeline shader stage info
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Assign the pipeline states to the pipeline creation info structure
		pipelineCreateInfo.pVertexInputState = &vertexInputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.renderPass = renderPass;
		pipelineCreateInfo.pDynamicState = &dynamicState;

		// Create rendering pipeline using the specified states
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &arena_pipeline));

		// Shader modules are no longer needed once the graphics pipeline has been created
		vkDestroyShaderModule(device, shaderStages[0].module, nullptr);
		vkDestroyShaderModule(device, shaderStages[1].module, nullptr);
	}

	void arena_prepareUniformBuffers()
	{
		// Prepare and initialize a uniform buffer block containing shader uniforms
		// Single uniforms like in OpenGL are no longer present in Vulkan. All Shader uniforms are passed via uniform buffer blocks
		VkMemoryRequirements memReqs;

		// Vertex shader uniform buffer block
		VkBufferCreateInfo bufferInfo = {};
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.pNext = nullptr;
		allocInfo.allocationSize = 0;
		allocInfo.memoryTypeIndex = 0;

		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(arena_uboVS);
		// This buffer will be used as a uniform buffer
		bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

		// Create a new buffer
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &arena_uniformBufferVS.buffer));
		// Get memory requirements including size, alignment and memory type 
		vkGetBufferMemoryRequirements(device, arena_uniformBufferVS.buffer, &memReqs);
		allocInfo.allocationSize = memReqs.size;
		// Get the memory type index that supports host visibile memory access
		// Most implementations offer multiple memory types and selecting the correct one to allocate memory from is crucial
		// We also want the buffer to be host coherent so we don't have to flush (or sync after every update.
		// Note: This may affect performance so you might not want to do this in a real world application that updates buffers on a regular base
		allocInfo.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		// Allocate memory for the uniform buffer
		VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &(arena_uniformBufferVS.memory)));
		// Bind memory to buffer
		VK_CHECK_RESULT(vkBindBufferMemory(device, arena_uniformBufferVS.buffer, arena_uniformBufferVS.memory, 0));
		
		// Store information in the uniform's descriptor that is used by the descriptor set
    arena_uniformBufferVS.descriptor.buffer = arena_uniformBufferVS.buffer;
    arena_uniformBufferVS.descriptor.offset = 0;
    arena_uniformBufferVS.descriptor.range = sizeof(arena_uboVS);
    
    arena_updateUniformBuffers();
	}

	void arena_updateUniformBuffers()
	{
		// Update matrices
    arena_uboVS.projectionMatrix = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 1024.0f);

    /*
		uboVS.viewMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, zoom));
		uboVS.modelMatrix = glm::mat4(1.0f);
    uboVS.modelMatrix = glm::rotate(uboVS.modelMatrix, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    uboVS.modelMatrix = glm::rotate(uboVS.modelMatrix, glm::radians(-rotation.y), glm::vec3(0.0f, 0.0f, 1.0f));
    */

    arena_uboVS.viewMatrix = glm::mat4(1.0f);


    float x = zoom * cos(rotation.y/3000.f);
    float y = zoom * sin(rotation.y/3000.f);
    float z = 50.f;


    arena_uboVS.modelMatrix = glm::lookAt(glm::vec3(x, y, z), glm::vec3(0, 0, 0), glm::vec3(0, 0, -1.f));

		// Map uniform buffer and update it

    uint8_t *pData;
		
    VK_CHECK_RESULT(vkMapMemory(device, arena_uniformBufferVS.memory, 0, sizeof(arena_uboVS), 0, (void **)&pData));
		
    memcpy(pData, &arena_uboVS, sizeof(arena_uboVS));
		
    // Unmap after data has been copied
		// Note: Since we requested a host coherent memory type for the uniform buffer, the write is instantly visible to the GPU
		
    vkUnmapMemory(device, arena_uniformBufferVS.memory);
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		prepareSynchronizationPrimitives();
		arena_prepareVertices();
		arena_prepareUniformBuffers();
		arena_setupDescriptorSetLayout();
		arena_preparePipelines();
		setupDescriptorPool();
		arena_setupDescriptorSet();
		buildCommandBuffers();
    prepareTextOverlay();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
	}

	virtual void viewChanged()
	{
		// This function is called by the base example class each time the view is changed by user input
		arena_updateUniformBuffers();
    updateTextOverlay();
	}
};

// OS specific macros for the example main entry points
// Most of the code base is shared for the different supported operating systems, but stuff like message handling diffes

#if defined(_WIN32)
// Windows entry point
VulkanExample *vulkanExample;
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleMessages(hWnd, uMsg, wParam, lParam);
	}
	return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR pCmdLine, int nCmdShow)
{
	for (size_t i = 0; i < __argc; i++) { VulkanExample::args.push_back(__argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow(hInstance, WndProc);
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}

#elif defined(__ANDROID__)
// Android entry point
VulkanExample *vulkanExample;
void android_main(android_app* state)
{
	vulkanExample = new VulkanExample();
	state->userData = vulkanExample;
	state->onAppCmd = VulkanExample::handleAppCommand;
	state->onInputEvent = VulkanExample::handleAppInput;
	androidApp = state;
	vulkanExample->renderLoop();
	delete(vulkanExample);
}
#elif defined(_DIRECT2DISPLAY)

// Linux entry point with direct to display wsi
// Direct to Displays (D2D) is used on embedded platforms
VulkanExample *vulkanExample;
static void handleEvent()
{
}
int main(const int argc, const char *argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
VulkanExample *vulkanExample;
int main(const int argc, const char *argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif defined(__linux__)

// Linux entry point
VulkanExample *vulkanExample;
static void handleEvent(const xcb_generic_event_t *event)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleEvent(event);
	}
}
int main(const int argc, const char *argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#endif
