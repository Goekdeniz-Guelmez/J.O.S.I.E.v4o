### Project J.O.S.I.E.v4o Description

**Overview:**
J.O.S.I.E. (Just an Outstandingly Smart and Intelligent Entity) v4o is an advanced AI assistant designed to revolutionize both conversational AI and smart home management. Developed with cutting-edge multimodal capabilities, J.O.S.I.E. can interpret and respond to a variety of inputs including images, videos, thermal images, depth, and audio. This makes it exceptionally versatile in understanding and interacting with its environment and users.

J.O.S.I.E. serves two primary functions:

1. **Conversational General-Purpose AI Assistant:** 
   - Equipped with natural language processing (NLP) and natural language understanding (NLU), J.O.S.I.E. engages in meaningful and context-aware conversations.
   - It can provide information, perform tasks, answer questions, and assist with daily activities, leveraging vast knowledge bases and dynamic learning algorithms.

2. **Autonomous Smart Home Manager:** 
   - J.O.S.I.E. integrates seamlessly with smart home devices and systems, allowing for intuitive control and automation.
   - It can manage lighting, climate control, security systems, appliances, and more, enhancing home comfort, efficiency, and security.

**Smart Home Capabilities:**

- **Security Systems:**
  - Integrates with home security systems, including cameras, alarms, and smart locks.
  - Provides real-time monitoring and alerts, and can perform security checks or control access to the home.

**Prompt Template:**

```text
<|begin_of_text|>system
You are J.O.S.I.E. which is an acronym for "Just an Outstandingly Smart Intelligent Entity", a private and super-intelligent AI assistant, created by Gökdeniz Gülmez.<|end_of_text|>
<|begin_of_text|>main user "Gökdeniz Gülmez"
{{ .Prompt }}<|end_of_text|>
<|begin_of_text|>josie
{{ .Response }}<|end_of_text|>
```

**User Roles and Access:**

1. **Main User (Gökdeniz Gülmez):**
   - Full access to J.O.S.I.E.’s complete suite of capabilities, including comprehensive control over smart home functions.
   - Ability to update and manage user access levels and permissions.

2. **Authorized Users:**
   - Granted access to general-purpose conversational features.
   - Restricted from controlling or accessing smart home functionalities.
   - Identified and authenticated by name.

3. **Unauthorized Users:**
   - Identified by name if possible, or labeled as "unknown."
   - Completely restricted from accessing any of J.O.S.I.E.’s abilities.
   - Interactions are redirected to the main user or trigger predefined security measures.

This version is only trained with the main User Role.

**Security Measures:**
J.O.S.I.E. employs robust security protocols to safeguard against unauthorized access. This includes user verification methods, such as biometric authentication and secure password management, to ensure only authorized users can interact with sensitive functions.

**Future Enhancements:**
The development roadmap for J.O.S.I.E. includes ongoing refinement of its NLP and NLU capabilities, deeper integration with emerging smart home technologies, and enhanced AI learning mechanisms. These advancements aim to make J.O.S.I.E. an even more powerful and intuitive assistant, continually improving user experience and home automation efficiency.

**Conclusion:**
J.O.S.I.E. v4o is poised to set a new standard in AI assistant technology, combining sophisticated conversational abilities with comprehensive smart home management. This dual functionality, coupled with strong security measures, positions J.O.S.I.E. as an essential tool for a smart, efficient, and secure living environment.

### **Development Stages:**

1. **Current Stage (Beta2): Conversational AI**
   - At this stage, J.O.S.I.E. is primarily a conversational assistant, fine-tuned using a custom prompt template inspired by ChatML.
   - The current prompt template:

```text
<|begin_of_text|>system
You are J.O.S.I.E. which is an acronym for "Just an Outstandingly Smart Intelligent Entity", a private and super-intelligent AI assistant, created by Gökdeniz Gülmez.<|end_of_text|>
<|begin_of_text|>main user "Gökdeniz Gülmez"
{{ .Prompt }}<|end_of_text|>
<|begin_of_text|>josie
{{ .Response }}<|end_of_text|>
```
   - This template ensures that interactions are personalized for the main user, Gökdeniz Gülmez.

2. **Next Stage (Beta3): Long-Term Memory Integration**
   - The next development stage will introduce long-term memory, enabling J.O.S.I.E. to retain information about the user and provide more contextually relevant responses over time.
   - The updated prompt template will include a JSON object to store user information:

```text
<|begin_of_text|>system
You are J.O.S.I.E. which is an acronym for "Just an Outstandingly Smart Intelligent Entity", a private and super-intelligent AI assistant, created by Gökdeniz Gülmez.<|end_of_text|>
<|begin_of_text|>long term memory
{"name": "Gökdeniz Gülmez", "age": 24, ...}<|end_of_text|>
<|begin_of_text|>main user "Gökdeniz Gülmez"
{{ .Prompt }}<|end_of_text|>
<|begin_of_text|>josie
{{ .Response }}<|end_of_text|>
```

3. **Future Stage: Function Calling**
   - In the subsequent stage, J.O.S.I.E. will be enhanced with the ability to call external functions, integrating with various tools and APIs to perform complex tasks.
   - The expanded prompt template for function calling will look like this:

```text
<|begin_of_text|>system
You are J.O.S.I.E. which is an acronym for "Just an Outstandingly Smart Intelligent Entity", a private and super-intelligent AI assistant, created by Gökdeniz Gülmez.<|end_of_text|>
<|begin_of_text|>available tools
{ .Tools }<|end_of_text|>
<|begin_of_text|>long term memory
{"name": "Gökdeniz Gülmez", "age": 24, ...}<|end_of_text|>
<|begin_of_text|>main user "Gökdeniz Gülmez"
{{ .Prompt }}<|end_of_text|>
<|begin_of_text|>josie
{ "tool_call": {"name": "name_of_the_tool", ...} }<|end_of_text|>
<|begin_of_text|>tool response
{{ .Response }}<|end_of_text|>
<|begin_of_text|>josie
{{ .Response }}<|end_of_text|>
```


# Training Stages


## Stage 1:
LLM Pre-Training:
- Pre-Trains the LLM model with loads of text.
- Layers Frozen:
	- Input multimodal layers (Encoder).
	- Output  multimodal layers (Decoders).
- Layer Unfrozen:
	- LLM (Gemma, Mistral, Llama, ...).


## Stage 2
Fine-Tuning the LLM on conversational Data with the custom Prompt Format.
- Layers Frozen:
	- Encoders.
	- Decoders.
- Layers Unfrozen:
	- LLM


## Stage 3
Encoder-side alignment via image/video/audio:
- Trains Input Models to output the right Embeddings.
- Contrastive Learning.
- Layers Frozen:
	- LLM
	- Output  multimodal layers (Decoders).
- Layer Unfrozen:
	- Input Modalities (Encoders)
- To Train the encoders to output Embeddings of different Modalities to be close to the same  Embeddings the LLM has.
- Goes like this:
	- Step 1:
		- Get the Image, Audio, Video, ... Descriptions or Text Pairs of the modalities.
	- Step 2:
		- Run the Text Pairs through the Embeddings Table of the LLM. To get the wanted Embeddings.
	- Stage 3:
		- Train the Modality Transformers with Contrastive Learning. Keep in mind, the wanted Embeddings  are from the LLM with the Text pairs.
- Pipeline one training step **(Beta 1)**:
	- **Step 1**: Get text Pair from modality.
	- **Step 2**: Look the Text in the LLMs Embeddings Table.
	- **Step 3**: Get the associated modality.
	- **Step 4**: Forward Pass the modality through the Transformer-Encoder.
	- **Step 5**: Compare the output Embeddings from the Transformer-Encoder with the Embeddings from the LLM to minimize the Distance between them.


## Stage 4?
Decoder-side alignment:
- Trains the Decoder Diffusion models to output the wanted files.
- Layers Frozen:
	- Encoders
	- LLM
- Layer Unfrozen:
	- Decoders
- Or just use already Pre-Trained Models.

- If in Voice mode there is a seconf Text to Speech model.
    - Embeddings from the LLM go throug a Linear Layer and then get feed directly into the TTS model.
    - Get output Emgeddings from the LLM and the wanted Audio Output compare them and get the Loss fot Bacwards Propagation.


## Stage 5
Instruction-following training.
- Fine-Tunes the model to output the right token representations to the right Modalities.
- Uses LoRA Fine-Tuning.
- Layers Frozen:
	- Encoders.
	- Decoders.
	- Parts of the LLM.
- Layers Unfrozen:
	- Parts of the LLM.


