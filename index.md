---
layout: default
title: Deep Learning @ VU
---

{::nomarkdown}

<nav class="lectures">

<!-- NB: The course details should not be linked from the home page -->

<div class="l1"><a href="/introduction">
 <h3 class="blue"><span>1. </span>Introduction</h3>
</a>
</div>

<div class="l2"><a href="/backpropagation">
 <h3 class="green"><span>2. </span>Backpropagation</h3>
</a>
</div>

<div class="l3"><a href="/cnns">
 <h3 class="orange"><span>3. </span>Convolutions</h3>
</a>
</div>

<div class="l4"><a href="/tools">
 <h3 class="blue"><span>4. </span>Tools of the trade</h3>
</a>
</div>

<div class="l5"><a href="/sequences">
 <h3 class="orange"><span>5. </span>Sequences</h3>
</a>
</div>

<div class="l6"><a href="/vae">
 <h3 class="blue"><span>6. </span>Latent variable models</h3>
</a>
</div>

<div class="l7"><span class="/repr">
 <h3 class="red"><span>7. </span>Unsupervised representation learning</h3>
</span>
</div>

<div class="l8"><a href="/graphs">
 <h3 class="orange"><span>8. </span>Graphs</h3>
</a>
</div>

<div class="l9"><a href="/sa">
 <h3 class="blue"><span>9. </span>Self-attention</h3>
</a>
</div>

<div class="l10"><span class="/pdfs/lecture10.reinforcementlearning.pdf">
 <h3 class="red"><span>10. Reinforcement learning</span></h3>
</span>
</div>

<div class="l11"><a href="/diffusion">
 <h3 class="blue"><span>11. </span>Diffusion</h3>
</a>
</div>

<div class="l12"><span class="/generalization">
 <h3 class="red"><span>12. Generalization</span></h3>
</span>
</div>

<div class="l13"><span class="soon">
 <h3 class="red"><span>13. Explainability</span></h3>
</span>
</div>


</nav>

{:/nomarkdown}

## Content

{::nomarkdown}

<table class="overview">
	<colgroup>
		<col class="week">
		<col class="lecture">
		<col class="lecturer">
		<col class="links">
	</colgroup>

  <tr>
    <th></th>
    <th>Lecture</th>
    <th>Lecturer</th>
    <th>Links</th>
  </tr>
  
  <tr>

  <tr>
   <td rowspan="3">w1</td>
    <td>
      <h3>1. Introduction </h3>
      <ul class="videos">
      <li><a href="./introduction/#video-000">1.1 Neural networks</a></li>
      <li><a href="./introduction/#video-020">1.2 Classification and regression</a></li>
      <li><a href="./introduction/#video-034">1.3 Autoencoders</a></li>
      </ul>
    </td>
    <td>Peter Bloem</td>
    <td><a href="https://www.youtube.com/watch?v=MrZvXcwQJdg&list=PLIXJ-Sacf8u53w3iLVjYImNXcLAPV2Do_&ab_channel=DLVU">playlist</a><br><a href="./pdf/lecture01.introduction.annotated.pdf">pdf</a> 
  </tr>
  <tr>
    <td>
      <h3>2. Backpropagation </h3>
      <ul class="videos">
      <li><a href="./backpropagation/#video-004">2.1 Scalar backpropagation</a></li>
      <li><a href="./backpropagation/#video-042">2.2 Tensor backpropagation</a></li>
      <li><a href="./backpropagation/#video-066">2.3 Automatic differentiation</a></li>
      <li><a href="./backpropagation/#video-042">2.4 Tensor details*</a></li>
      </ul>
    </td>
    <td>Peter Bloem</td>
    <td><a href="https://www.youtube.com/watch?v=idO5r5eWIrw&list=PLIXJ-Sacf8u7YQ77QmD5rFgAlDgFLqZ4b&index=4&ab_channel=DLVU">playlist</a><br><a href="./pdfs/lecture02.backpropagation.annotated.pdf">pdf</a> 
  </tr>
  <tr>
    <td>
      <h3>3. Convolutions </h3>
      <ul class="videos">
      <li><a href="./cnns/#video-000">3.1 Introduction</a></li>
      <li><a href="./cnns/#video-016">3.2 Conv1D (a)</a></li>
      <li><a href="./cnns/#video-042">3.3 Conv1D (b)</a></li>
      <li><a href="./cnns/#video-060">3.3 Conv2D, Conv3D, ConvND</a></li>
      </ul>
    </td>
    <td>Michel Cochez</td>
    <td><a href="https://www.youtube.com/watch?v=rOuF5r5GduQ&list=PLIXJ-Sacf8u4koFI1FzdM6KYVDCLhaepZ&ab_channel=DLVU">playlist</a><br><a href="./pdfs/lecture03.cnns.annotated.pdf">pdf</a> 
  </tr>

  <tr>
   <td rowspan="3">w2</td>
    <td>
      <h3>3. Tools of the trade </h3>
      <ul class="videos">
      <li><a href="./tools/#video-003">4.1 Deep learning in practice</a></li>
      <li><a href="./tools/#video-044">4.2 Why does any of this work at all?</a></li>
      <li><a href="./tools/#video-066">4.3 Understanding optimizers</a></li>
      <li><a href="./tools/#video-101">4.4 The bag of tricks</a></li>
      </ul>
    </td>
    <td>Peter Bloem</td>
    <td><a href="https://www.youtube.com/playlist?list=PLIXJ-Sacf8u4XtBpteHSsW9j0WCx8MYbv">playlist</a><br><a href="./pdfs/lecture04.tools.annotated.pdf">pdf</a> 
  </tr>
  <tr>
    <td>
      <h3>5. Sequences </h3>
      <ul class="videos">
      <li><a href="./sequences/#video-002">5.1 Learning from sequences</a></li>
      <li><a href="./sequences/#video-042">5.2 Recurrent neural networks</a></li>
      <li><a href="./sequences/#video-060">5.3 LSTMs and friends</a></li>
      <li><a href="./sequences/#video-084">5.4 CNNs for sequential data</a></li>
      <li><a href="./sequences/#video-102">5.5 ELMo,  a case study</a></li>
      </ul>
    </td>
    <td>Peter Bloem, David Romero</td>
    <td><a href="https://www.youtube.com/playlist?list=PLIXJ-Sacf8u4koFI1FzdM6KYVDCLhaepZ">playlist</a><br><a href="./pdfs/lecture05.sequences.annotated.pdf">pdf</a> 
  </tr>

  <tr>
   <td rowspan="3">w3</td>
    <td>
      <h3>6. Latent Variable Models </h3>
      <ul class="videos">
      <li><a href="./vae/#video-002">6.1 Why Generative Modeling</a></li>
      <li><a href="./vae/#video-014">6.2 Autoencoders</a></li>
      <li><a href="./vae/#video-023">6.3 Variational Autoencoders</a></li>
      </ul>
    </td>
    <td>Shujian Yu</td>
    <td><a href="./pdfs/lecture06.latentvariablemodels.annotated.pdf">pdf</a> 
  </tr>
  <tr>
    <td>
      <h3>7. Unsupervised representation learning </h3>
      <ul class="videos">
      <li><a href="./repr/#video-000">7.0 Introduction</a></li>
      <li><a href="./repr/#video-008">7.1 VAE Implementation</a></li>
      <li><a href="./repr/#video-023">7.2 KL Divergence</a></li>
      <li><a href="./repr/#video-035">7.3 MMD-VAE</a></li>
      </ul>
    </td>
    <td>Shujian Yu</td>
    <td><a href="./pdfs/lecture07.UnsupervisedRepresentation.unannotated.pdf">pdf</a> 
  </tr>

  <tr>
   <td rowspan="3">w4</td>
    <td>
      <h3>8. Learning with graphs </h3>
      <ul class="videos">
      <li><a href="./graphs/#video-002">8.1 Introduction - Graphs (1A)</a></li>
      <li><a href="./graphs/#video-017">8.2 Introduction - Embeddings (1B)</a></li>
      <li><a href="./graphs/#video-025">8.3 Graph Embedding Techniques</a></li>
      <li><a href="./graphs/#video-048">8.4 Graph Neural Networks</a></li>
      <li><a href="./graphs/#video-064">8.5 Query embedding</a></li>
      </ul>
    </td>
    <td>Michael Cochez</td>
    <td><a href="https://www.youtube.com/playlist?list=PLIXJ-Sacf8u5IU-oyWn5bwF6c8XcR1TAR">playlist</a><br><a href="./pdfs/lecture08.graphs.annotated.pdf">pdf</a> 
  </tr>
  <tr>
    <td>
      <h3>9. Transformers and self-attention </h3>
      <ul class="videos">
      <li><a href="./sa/#video-002">9.1 Self-attention</a></li>
      <li><a href="./sa/#video-028">9.2 Transformers</a></li>
      <li><a href="./sa/#video-048">9.3 Famous transformers</a></li>
      <li><a href="./sa/#video-072">9.4 Scaling up</a></li>
      </ul>
    </td>
    <td>Peter Bloem</td>
    <td><a href="https://www.youtube.com/playlist?list=PLIXJ-Sacf8u7UwAsGYFR1952pzQux5Lb1">playlist</a><br><a href="./pdfs/lecture09.self-attention.annotated.pdf">pdf</a> 
  </tr>

  <tr>
   <td rowspan="3">w5</td>
    <td>
      <h3>10. Reinforcement learning </h3>
    </td>
    <td>Vincent Francois-Lavet</td>
    <td><a href="./pdfs/lecture10.reinforcementlearning.pdf">pdf</a> 
   </tr>
  </tr>

  <tr>
   <td rowspan="3">w6</td>
    <td>
      <h3>11. Diffusion models </h3>
      <ul class="videos">
      <li><a href="./diffusion/#video-005">11.1 Naive diffusion</a></li>
      <li><a href="./diffusion/#video-022">11.2 Understanding Gaussians</a></li>
      <li><a href="./diffusion/#video-049">11.3 Gaussian diffusion</a></li>
      </ul>
    </td>
    <td>Peter Bloem</td>
    <td><a href="https://www.youtube.com/watch?v=mCVkLU2x4xY&list=PLIXJ-Sacf8u4Nq3vmR1Nde9UlKiPaecEA&pp=iAQB">playlist</a><br><a href="./pdfs/lecture11.diffusion.annotated.pdf">pdf</a> 
  </tr>
  <tr>
    <td>
      <h3>12. Generalization </h3>
      <ul class="videos">
      <li><a href="./generalization/#video-000">12.0 Introduction</a></li>
      <li><a href="./generalization/#video-002">12.1 Review</a></li>
      <li><a href="./generalization/#video-017">12.2 Problem</a></li>
      <li><a href="./generalization/#video-028">12.3 Generalization Bound</a></li>
      </ul>
    </td>
    <td>Shujian Yu</td>
    <td><a href="./pdfs/lecture12.generalization.pdf">pdf</a> 
  </tr>

</table>

{:/nomarkdown}

## Last year's content

<table>
  <tr>
   <th></th>
    <th></th>
    <th>lecturer</th>
    <th>videos</th>
    <th>slides</th>
  </tr>
  <tr>
    <td>week 1</td> 
    <td>Introduction</td> 
    <td>Jakub Tomczak</td>
    <td>
    <a class=" inline_disabled" href="https://youtu.be/vTyZH8oqTec">A</a>, <a class=" inline_disabled" href="https://youtu.be/i7-nhWSFsZ8">B</a>, <a class=" inline_disabled" href="https://youtu.be/uk3TGBQqMtU">C</a>, <a class=" inline_disabled" href="https://youtu.be/I5lJ7Z-rL1A">D</a> 
    </td> 
    <td>
    <a href="/slides/dlvu.lecture01.pdf">pdf</a>
    </td>
  </tr>
  <tr>
    <td></td>
    <td>Backpropagation</td> 
    <td>Peter Bloem</td>
  <td>
    <a class=" inline_disabled" href="https://youtu.be/COhjLwjEpGM">A</a>, <a class=" inline_disabled" href="https://youtu.be/7mTcWrnexkk">B</a>, <a class=" inline_disabled" href="https://youtu.be/dxZ8a-oIu7U">C</a>, <a class=" inline_disabled" href="https://youtu.be/UpLtbV4L6PI">D</a>
    </td> 
    <td>    <a href="/slides/dlvu.lecture02.pdf">pdf</a></td>
  </tr>
  <tr>
    <td></td> 
	<td>Convolutional Neural Networks</td> 
    <td>Michael Cochez</td>	
    <td>
    <a class=" inline_disabled" href="https://youtu.be/rOuF5r5GduQ">A</a>, <a class=" inline_disabled" href="https://youtu.be/VQqayqUCTwM">B</a>, <a class=" inline_disabled" href="https://youtu.be/Q7KekwUricc">C</a>, <a class=" inline_disabled" href="https://youtu.be/2hS_54kgMHs">D</a>
    </td> 
    <td>    <a href="/slides/dlvu.lecture03.pdf">pdf</a></td>
  </tr>
 <tr>
    <td>week 2 </td>
    <td>Sequential data</td>
    <td>Peter Bloem</td>
    <td>
    <a class=" inline_disabled" href="https://youtu.be/rK20XfDN1N4">A</a>, <a class=" inline_disabled" href="https://youtu.be/2JGlmBhQedk">B</a>, <a class=" inline_disabled" href="https://youtu.be/fbTCvvICk8M">C</a>, <a class=" inline_disabled" href="https://www.youtube.com/watch?v=rT77lBfAZm4&amp;ab_channel=DLVU">D</a>, <a class=" inline_disabled" href="https://youtu.be/csAlW9HmwAQ">E</a>*
    </td> 
    <td>    <a href="/slides/dlvu.lecture05.pdf">pdf</a></td>
  </tr>
  <tr>
    <td></td>    
    <td>Tools of the trade</td> 
	<td>Peter Bloem</td>
<td>
    <a class=" inline_disabled" href="https://youtu.be/EE5jTGP7wrM">A</a>, <a class=" inline_disabled" href="https://youtu.be/ixI83iX7TV4">B</a>, <a class=" inline_disabled" href="https://youtu.be/uEvvs2YCxQk">C</a>, <a class=" inline_disabled" href="https://youtu.be/mX92C0s0q1Y">D</a>
    </td> 
    <td>    <a href="/slides/dlvu.lecture04.pdf">pdf</a></td>
  </tr>
  <tr>
    <td>week 3 </td> 
    <td>Latent Variable Models (pPCA and VAE)</td> 
    <td>Jakub Tomczak</td>
    <td>
    <a class=" inline_disabled" href="https://youtu.be/EfOZQvSCDsE">A</a>, <a class=" inline_disabled" href="https://youtu.be/BTUehwU_5Uo">B</a>, <a class=" inline_disabled" href="https://youtu.be/ywNkaCdr6nA">C</a>
    </td> 
    <td>    <a href="/slides/dlvu.lecture06.pdf">pdf</a></td>
  </tr>
  <tr>
    <td></td> 
    <td>GANs</td> 
    <td>Jakub Tomczak</td>
    <td>
    <a class=" inline_disabled" href="https://youtu.be/2nqtz3GzybQ">A</a>, <a class=" inline_disabled" href="https://youtu.be/Ydk-GqUMQQM">B</a>
    </td> 
    <td>    <a href="/slides/dlvu.lecture07.pdf">pdf</a></td>
  </tr>
  <tr>
    <td>week 4</td> 
    <td>Learning with Graphs</td>
	    <td>Michael Cochez</td>
    <td>
    <a href="https://www.youtube.com/playlist?list=PLIXJ-Sacf8u5IU-oyWn5bwF6c8XcR1TAR">playlist</a>
    </td> 
    <td>    <a href="/slides/dlvu.lecture08.pdf">pdf</a></td>
  </tr>
  <tr>
    <td></td>
    <td>Transformers & self-attention</td> 
    <td>Peter Bloem</td>
    <td><a class=" inline_disabled" href="https://youtu.be/KmAISyVvE1Y">A</a>, <a class=" inline_disabled" href="https://youtu.be/oUhGZMCTHtI">B</a>, <a class=" inline_disabled" href="https://youtu.be/MN__lSncZBs">C</a></td> 
    <td><a href="/slides/dlvu.lecture12.pdf">pdf</a></td>
  </tr>
<tr>
    <td>week 5</td> 
    <td>Reinforcement learning</td> 
	    <td>Emile van Krieken</td>
    <td>
    <a class=" inline_disabled" href="https://www.youtube.com/watch?v=t1I4NQTRXA0">A</a>, <a class=" inline_disabled" href="https://www.youtube.com/watch?v=6KzJ1bpcNC4">B</a>, <a class=" inline_disabled" href="https://www.youtube.com/watch?v=PikByfX0p80">C</a>
    </td> 
    <td>    <a href="/slides/dlvu.lecture09.pdf">pdf</a></td>
  </tr>
  <tr>
    <td></td>
	<td>Reinforcement learning (extra)</td>
    <td>Emile van Krieken</td> 
    <td>
    <a class=" inline_disabled" href="https://www.youtube.com/watch?v=mCVkLU2x4xY">A</a>, <a class=" inline_disabled" href="https://www.youtube.com/watch?v=ItI_gMuT5hw">B</a>, <a class=" inline_disabled" href="https://www.youtube.com/watch?v=zNCq1r4qI4Q">C</a>
    </td> 
    <td>    <a href="/slides/dlvu.lecture11.pdf">pdf</a></td>
  </tr>
  <tr>
    <td></td>
	<td>Autoregressive and Flow-based models</td> 
	<td>Jakub Tomczak</td>
    <td>
    <a class=" inline_disabled" href="https://youtu.be/_VPnu55UMCk">A</a>, <a class=" inline_disabled" href="https://youtu.be/d_h6kY0s9yI">B</a>, <a class=" inline_disabled" href="https://youtu.be/Rhx6W3dGvK8">C</a>
    </td> 
    <td>    <a href="/slides/dlvu.lecture10.pdf">pdf</a></td>
  </tr>
</table>