---
title: "test"
slides: true
---
<nav class="menu">
    <ul>
        <li class="home"><a href="/">Home</a></li>
        <li class="name">test</li>
        <li class="pdf"><a href="test_test">PDF</a></li>
    </ul>
</nav>

<article class="slides">



       <section id="slide-001">
            <a class="slide-link" href="test.com#slide-001" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0001.svg" class="slide-image" />

            <figcaption>
            <p    >Today’s lecture will be entirely devoted to the backpropagation algorithm. The heart of all deep learning. </p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-002">
            <a class="slide-link" href="test.com#slide-002" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0002.svg" class="slide-image" />

            <figcaption>
            <p    >In <strong>the first part</strong>, we will review the basics of neural networks. The rest of the lecture will mostly be about backpropagation: the algorithm that allows us to efficiently compute a gradient for the parameters of a neural net, so that we can train it using the gradient descent algorithm. The introductory lecture covered some of this material alreayd, but we'll go over the basics again to set up the notations and visualization for the rest of the lecture.<br></p><p    >In <strong>the second part</strong> we describe backpropagation in a <em>scalar</em> setting. That is, we will treat each individual element of the neural network as a single number, and simply loop over all these numbers to do backpropagation over the whole network. This simplifies the derivation, but it is ultimately a slow algorithm with a complicated notation.<br></p><p    >In <strong>the second part</strong>, we translate neural networks to operations on vectors, matrices and tensors. This allows us to simplify our notation, and more importantly, massively speed up the computation of neural networks. Backpropagation on tensors is a little more difficult to do than backpropagation on scalars, but it's well worth the effort.<br></p><p    >In <strong>the third part</strong>, we will make the final leap from manually worked out and implemented backpropagation system to  full-fledged <em>automatic differentiation</em>: we will show you how to build a system that takes care of the gradient computation entirely by itself. This is the technology behind software like pytorch and tensorflow.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-003">
            <a class="slide-link" href="test.com#slide-003" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0003.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-004">
            <a class="slide-link" href="test.com#slide-004" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0004.svg" class="slide-image" />

            <figcaption>
            <p    >We’ll start with a quick recap of the basic principles behind neural networks. The name comes from <em>neurons </em>the cells that make up most of our brain and nervous system.<br></p><p    >In the very early days of AI (the late 1950s), researchers decided to take a simple approach to AI. To simply model neurons in the computer. A neuron receives multiple different signals from other cells through connections called <strong>dendrites</strong>. It processes these in a relatively simple way, deriving a<em> single</em> new signal, which it sends out through its single <strong>axon</strong>. The axon branches out so that the single signal can reach multiple other cells.<br></p><p    ><br></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-005">
            <a class="slide-link" href="test.com#slide-005" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0005.svg" class="slide-image" />

            <figcaption>
            <p    >The of idea of a neuron needed to be radically simplified to work with computers of that age, but doing so yielded one of the first successful machine learning systems: <strong>the perceptron</strong>.<br></p><p    >The perceptron has a number of <em>inputs</em>, each of which is multiplied by a <strong>weight</strong>. The result is summed over all weights and inputs, together with a <strong>bias parameter</strong>, to provide the <em>output</em> of the perceptron. If we're doing binary classification, we can take the sign of the output as the class (if the output is bigger than 0 we predict class A otherwise class B).<br></p><p    >The bias parameter is often represented as a special input node, called a<strong> bias node</strong>, whose value is fixed to 1.<br></p><p    >For most of you, this will be nothing new. This is simply a linear classifier or linear regression model. It just happens to be drawn as a network. <br></p><p    >But the real power of the brain doesn't come from single neurons, it comes from <em>chaining a large number of neurons together</em>. Can we do the same thing with perceptrons: link the outputs of one perceptron to the inputs of the next in a large network, and so make the whole more powerful than any single perceptron?</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-006" class="anim">
            <a class="slide-link" href="test.com#slide-006" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0006_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0006_animation_0.svg,lecture_02.backpropagation.key-stage-0006_animation_1.svg" class="slide-image" />

            <figcaption>
            <p    >This is where the perceptron turns out to be too simple an abstraction. Composing perceptrons (making the output of one perceptron the input of another) doesn’t make them more powerful. All you end out with is something that is equivalent to another linear model. We’re not creating models that can learning non-linear functions.<br></p><aside    >We’ve removed the bias node here for clarity, but that doesn’t affect our conclusions: any composition of affine functions is itself an affine function.<br></aside><p    >If we're going to build networks of perceptrons that do anything a single perceptron can't do, we need another trick.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-007">
            <a class="slide-link" href="test.com#slide-007" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0007.svg" class="slide-image" />

            <figcaption>
            <p    >The simplest solution is to apply a <em>nonlinear</em> function to each neuron, called the <strong>activation function.</strong> This is a scalar function we apply to the output of a perceptron after all the weighted inputs have been combined. <br></p><p    >One popular option (especially in the early days) is the <strong>logistic sigmoid</strong>, which we’ve seen already. Applying a sigmoid means that the sum of the inputs can range from negative infinity to positive infinity, but the output is always in the interval [0, 1].<br></p><p    >Another, more recent nonlinearity is the<strong> linear rectifier</strong>, or <strong>ReLU</strong> nonlinearity. This function just sets every negative input to zero, and keeps everything else the same.<br></p><p    >Not using an activation function is also called using a <strong>linear activation</strong>.<br></p><aside    >If you're familiar with logistic regression, you've seen the sigmoid function already: it's stuck on the end of a linear regression function (that is, a perceptron) to turn the outputs into class probabilities. Now, we will take these sigmoid outputs, and feed them as inputs to other perceptrons.</aside><aside    ></aside>
            </figcaption>
       </section>





       <section id="slide-008">
            <a class="slide-link" href="test.com#slide-008" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0008.svg" class="slide-image" />

            <figcaption>
            <p    >Using these nonlinearities, we can arrange single neutrons into <strong>neural networks</strong>. Any arrangement makes a neural network, but for ease of training, the arrangement shown here was the most popular for a long time. It’s called a <strong>feedforward network </strong>or <strong>multilayer perceptron</strong>. We arrange a layer of hidden units in the middle, each of which acts as a perceptron with a nonlinearity, connecting to all input nodes. Then we have one or more output nodes, connecting to all hidden layers. Crucially:<br></p><p     class="list-item">There are <strong>no cycles</strong>, the network feeds forward from input to output.<br></p><p     class="list-item">Nodes in the same layer are not connected to  each other, or to any other layer than the previous one.<br></p><p     class="list-item">Each layer is <strong>fully connected</strong> to the previous layer, every node in one layer connects to every node in the layer before it.<br></p><p    >In the 80s and 90s feedforward networks usually had just one hidden layer, because we hadn’t figured out how to train deeper networks.<br></p><p    >Note: Every <span>orange </span>and <span>blue</span> line in this picture represents one parameter of the model.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-009">
            <a class="slide-link" href="test.com#slide-009" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0009.svg" class="slide-image" />

            <figcaption>
            <p    >If we want to train a regression model (a model that predicts a numeric value), we put non-linearities on the hidden nodes, and no activation on the output node. That way, the output can range from negative to positive infinity, and the nonlinearities on the hidden layer ensure that we can learn functions that a single perceptron couldn't learn (we can learn non-linear functions).<br></p><p    >We can think of the first layer as learning some nonlinear transformation of the features, and the second layer as performing linear regression on these derived  features.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-010" class="anim">
            <a class="slide-link" href="test.com#slide-010" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0010_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0010_animation_0.svg,lecture_02.backpropagation.key-stage-0010_animation_1.svg" class="slide-image" />

            <figcaption>
            <p    >If we have a classification problem with two classes, called positive and negative, we can place a sigmoid activation on the output layer, so that the output is between 0 and 1. We can then interpret this as the<strong> probability </strong>that the input has the <strong>positive</strong> class (according to our network). The probability of the negative class is 1 minus this value.<br></p><p    ><br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-011">
            <a class="slide-link" href="test.com#slide-011" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0011.svg" class="slide-image" />

            <figcaption>
            <p    >For multi-class classification, we can use the <strong>softmax activation</strong>. We create a single output node for every class, and ensure that their values sum to one. We can then interpret this series of values as the class probabilities that our network predicts. The softmax activation ensures positive values that sum to one.<br></p><p    >After the softmax we can interpret the output of node y<sub>3</sub> as the probability that our input has class 3.<br></p><p    >To compute the softmax, we simply take the exponent of each output node o<sub>i</sub> (to ensure that they are all positive) and then divide each by the total (to ensure that they sum to one). We could make the values positive in many other ways (taking the absolute or the square), but the exponent is a common choice for this sort of thing in statistics and physics, and it seems to work well enough.<br></p><p    >The softmax activation is a little unusual: it’s not <em>element-wise </em>like the sigmoid or the ReLU. To compute the value of one output node, it looks at the inputs of  all the other outputs nodes. <br></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-012" class="anim">
            <a class="slide-link" href="test.com#slide-012" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0012_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0012_animation_0.svg,lecture_02.backpropagation.key-stage-0012_animation_1.svg,lecture_02.backpropagation.key-stage-0012_animation_2.svg,lecture_02.backpropagation.key-stage-0012_animation_3.svg" class="slide-image" />

            <figcaption>
            <p    >Now that we know how to build a neural network, and how to compute its output for a given input, the next question is <em>how do we train it? </em>Given a particular network and a set of examples of which inputs correspond to which outputs, how do we find the weights for which the network makes good predictions?<br></p><p    >To find good weights we first define a<strong> loss function</strong>. This is a function of a particular model (represented by its<span> weights</span>) to a single scalar value, called the model's <strong>loss</strong>. The better our model, the lower the loss. If we imagine a model with just two weights then the set of all models, called the <strong>model space</strong>, forms a plane. For every point in this plane, our loss function defines a loss. We can draw this above the plane as a surface: the <strong>loss surface</strong><strong> </strong>(sometimes also called, more poetically, the loss landscape).<br></p><aside    >Shown in the slide is a simple example of a loss function (the Euclidean distance between the model output and the target). There are many other loss functions available, and we will see the most important ones throughout the course.<br></aside><p    >Our job is to search the loss surface for a low point. When the loss is low, the model predictions are close to the target labels, and we've found a model that does well.<br></p><aside    >Make sure you understand the difference between the model, a function from the inputs x to the outputs y in which the weights act as constants, and the loss function, a function from the weights to a loss value, <strong>in which the data acts as constants</strong>.<br></aside><p    >The symbol <strong>θ</strong> is a common notation referring to the set of all weights of a model (sometimes combined into a vector, sometimes just a set).</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-013">
            <a class="slide-link" href="test.com#slide-013" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0013.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-014">
            <a class="slide-link" href="test.com#slide-014" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0014.svg" class="slide-image" />

            <figcaption>
            <p    >This is a common way of summarizing the aim of machine learning. We have a large space of possible parameters, with <strong>θ </strong>representing a single choice, and in this space we want to find the <strong>θ </strong>for which the loss on our chosen dataset is minimized.<br></p><p    >It turns out this is actually an oversimplification, and we don't want to solve this particular problem <em>too</em> well. We'll discuss this in the fourth lecture. For now, this serves as a reasonable summary of what we're trying to do.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-015" class="anim">
            <a class="slide-link" href="test.com#slide-015" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0015_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0015_animation_0.svg,lecture_02.backpropagation.key-stage-0015_animation_1.svg,lecture_02.backpropagation.key-stage-0015_animation_2.svg,lecture_02.backpropagation.key-stage-0015_animation_3.svg,lecture_02.backpropagation.key-stage-0015_animation_4.svg,lecture_02.backpropagation.key-stage-0015_animation_5.svg" class="slide-image" />

            <figcaption>
            <p    >Here are some common loss functions for situations where we have examples (t) of what the model output (y) should be for a given input (x). <br></p><p    >The squared error losses are derived from basic regression. The (binary) cross entropy comes from logistic regression (as shown last lecture) and the hinge loss comes from support vector machine classification. You can find their derivations in most machine learning books/courses. We won’t elaborate on them here, except to note that in all cases the loss is lower if the model output (y) is closer to the example output (t).<br></p><p    >The loss can be computed for a single example or for multiple examples.<strong> In almost all cases, the loss for multiple examples is just the sum over all their individual losses.</strong></p><p    ><strong></strong></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-016" class="anim">
            <a class="slide-link" href="test.com#slide-016" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0016_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0016_animation_0.svg,lecture_02.backpropagation.key-stage-0016_animation_1.svg,lecture_02.backpropagation.key-stage-0016_animation_2.svg" class="slide-image" />

            <figcaption>
            <p    >In one dimension, we know that the derivative of a function (like the loss) tells us how much a function increases or decreases if we take a step of size 1 to the right.<br></p><aside    >To be more precise, it tells us how much the best linear approximation of the function at a particular point, the <span>tangent line</span>, increases or decreases.</aside><aside    ></aside>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-017" class="anim">
            <a class="slide-link" href="test.com#slide-017" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0017_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0017_animation_0.svg,lecture_02.backpropagation.key-stage-0017_animation_1.svg,lecture_02.backpropagation.key-stage-0017_animation_2.svg,lecture_02.backpropagation.key-stage-0017_animation_3.svg" class="slide-image" />

            <figcaption>
            <p    >If our input space has multiple dimensions, like our model space, we can simply take the derivative with respect to each input, separately, treating the others as constants. This is called a<strong> partial derivative</strong>. The collection of all possible partial derivatives is called <strong>the gradient</strong>. <br></p><p    >If we interpret the gradient as a vector, it points in the direction in which the function grows the fastest. Taking a step in the opposite direction means we are walking down the function.<br></p><p    >In our case, this means that if we can work out the gradient of the loss (which contains the model and the data), then we can take a small step in the opposite direction and be sure that we are moving to a lower point on the loss surface.<br></p><p    >The symbol for the gradient is a downward pointing triangle called a nabla. The subscript indicates the variable over which we are taking the derivatives.  Note that in this case we are treating <span>θ</span> as a vector.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-018">
            <a class="slide-link" href="test.com#slide-018" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0018.svg" class="slide-image" />

            <figcaption>
            <p    >Here's a simple way to think of the gradient. <strong>The gradient is an arrow that points in the direction of steepest ascent</strong>. That is, the gradient of our loss (at the dot) is the direction in which the loss surface increases the quickest. <br></p><p    >More precisely, if we fit a tangent hyperplane to the loss surface at the dot, then the direction of steepest ascent on that hyperplane is the gradient. Since it's a hyperplane, the opposite direction (the gradent with a minus in front of it) is the direction of steepest descent.<br></p><p    >This is why we care about the gradent: it helps us find a downward direction on the loss surface. All we have to do is follow the negative gradient and we will end up lowering our loss.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-019" class="anim">
            <a class="slide-link" href="test.com#slide-019" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0019_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0019_animation_0.svg,lecture_02.backpropagation.key-stage-0019_animation_1.svg" class="slide-image" />

            <figcaption>
            <p    >This is the idea behind the <strong>gradient descent algorithm</strong>. We compute the gradient, take a small step in the opposite direction and repeat. The reason we take small steps is that the gradient is only the direction of steepest ascent <em>locally</em>. It's a linear approximation to the nonlinear loss function. The further we move from our current position the worse an approximation the tangent hyperplane will be for the function that we are actually trying to follow. That's why we only take a small step, and then <em>recompute</em> the gradient in our new position.<br></p><p    >We can compute the gradient of the loss with respect to a single example from our dataset, a small batch of examples, or over the whole dataset. These options are usually called stochastic, minibatch and full-batch gradient descent respectively (although minibatch and stochastic gradient descent are sometimes used interchangeably).<br></p><p    >In deep learning, we almost always use <strong>minibatch gradient descent,</strong> but there are some cases where full batch is used.<br></p><p    >Training usually requires multiple passes over the data. One such pass is called an <strong>epoch</strong>.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-020">
            <a class="slide-link" href="test.com#slide-020" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0020.svg" class="slide-image" />

            <figcaption>
            <p    >This is the basic idea of neural network training. What we haven't discussed is how to work out the gradient of a loss function over a neural network. For simple functions like linear classifiers, this can be done by hand. For more complex functions, like very deep neural networks, this is no longer feasible, and we need some help. <br></p><p    >This help comes in the form of the <strong>backpropagation</strong> algorithm.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-021">
            <a class="slide-link" href="test.com#slide-021" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0021.svg" class="slide-image" />

            <figcaption>
            <aside    >Before we move on, it's important to note that the name neural network is not much more than a historical artifact. The original neural neural networks were very loosely inspired by thenetworks of neurons in our heads, but even then the artificial neural nets were so simplified that they had little to do with the real thing. Today's neural nets are nothing like brain networks, and serve in no way as a realistic model of what happens in our head. In short, don't read too much into the name.<br></aside><p    ><br></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-022">
            <a class="slide-link" href="test.com#slide-022" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0025.svg" class="slide-image" />

            <figcaption>
            <p    >In the previous video, we looked at what neural networks are, and we saw that to train them, we need to work out the derivatives of the loss with respect to the parameters of the neural network: collectively these derivatives are known as <em>the gradient</em>.<br></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-023">
            <a class="slide-link" href="test.com#slide-023" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0026.svg" class="slide-image" />

            <figcaption>
            <p    >For simple models, like logistic regression, working out a gradient is usually done by hand, with pen and paper. <br></p><p    >This function is then translated into code, and used in a gradient descent loop. But the more complicated our model becomes, the more complex it becomes to work out a complete formulation of the gradient.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-024">
            <a class="slide-link" href="test.com#slide-024" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0027.svg" class="slide-image" />

            <figcaption>
            <p    >Here is a diagram of the sort of network we’ll be encountering (this one is called the GoogLeNet). We can’t work out a complete gradient for this kind of architecture by hand. We need help. What we want is some sort of algorithm that lets the computer work out the gradient for us.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-025">
            <a class="slide-link" href="test.com#slide-025" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0028.svg" class="slide-image" />

            <figcaption>
            <p    >Of course, working out derivatives is a pretty mechanical process. We could easily take all the rules we know, and put them into some algorithm. This is called <strong>symbolic differentiation</strong>, and it’s what systems like Mathematica and Wolfram Alpha do for us.<br></p><p    >Unfortunately, as you can see here, the derivatives it returns get pretty horrendous the deeper the neural network gets. This approach becomes impractical very quickly. <br></p><aside    >As the depth of the network grows, the symbolic expression of its gradient (usually) grows exponentially in size.<br></aside><p    >Note that in symbolic differentiation we get a description of the derivative that<strong> is independent of the input</strong>. We get a function that we can then feed any input to.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-026">
            <a class="slide-link" href="test.com#slide-026" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0029.svg" class="slide-image" />

            <figcaption>
            <p    >Another approach is to compute the gradient <em>numerically</em>. For instance by the method of finite differences: we take a small step<span> ε </span>and, see how much the function changes. The amount of change divided by the step size is a good estimate for the  gradient if <span>ε</span> is small enough.<br></p><p    >Numeric approaches are sometimes used in deep learning, but it’s very expensive to make them accurate enough if you have a large number of parameters.<br></p><p    >Note that in the numeric approach, you only get an answer <strong>for a particular input</strong>. If you want to compute the gradient at some other point in space, you have to compute another numeric approximation. Compare this to the symbolic approach (either with pen and paper or through wolfram alpha) where once the differentation is done, all we have to compute is the derivative that we've worked out.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-027">
            <a class="slide-link" href="test.com#slide-027" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0030.svg" class="slide-image" />

            <figcaption>
            <p    >Backpropagation is a kind of middle ground between symbolic and numeric approaches to working out the gradient. We break the computation into parts: we work out the derivatives of the parts <em>symbolically</em>, and then chain these together <em>numerically</em>.<br></p><p    >The secret ingredient that allows us to make this work is the <strong>chain rule</strong> of differentiation.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-028" class="anim">
            <a class="slide-link" href="test.com#slide-028" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0031_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0031_animation_0.svg,lecture_02.backpropagation.key-stage-0031_animation_1.svg,lecture_02.backpropagation.key-stage-0031_animation_2.svg,lecture_02.backpropagation.key-stage-0031_animation_3.svg" class="slide-image" />

            <figcaption>
            <p    >Here is the chain rule: if we want the derivative of a function which is the composition of two other functions, in this case <span>f</span> and <span>g</span>, we can take the derivative of <span>f</span> with respect to the output of <span>g</span> and  multiply it by the derivative of <span>g</span> with respect to the input x.<br></p><p    >Since we’ll be using the chain rule <em>a lot</em>, we’ll introduce a simple shorthand to make it a little easier to parse. We draw a little diagram of which function feeds into which. This means we know what the argument of each function is, so we can remove the arguments from our notation. <br></p><p    >We call this diagram a <strong>computation graph</strong>. We'll stick with simple diagrams like this for now. At the end of the lecture, we will expand our notation a little bit to capture more detail of the computation.<br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-029" class="anim">
            <a class="slide-link" href="test.com#slide-029" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0032_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0032_animation_0.svg,lecture_02.backpropagation.key-stage-0032_animation_1.svg,lecture_02.backpropagation.key-stage-0032_animation_2.svg,lecture_02.backpropagation.key-stage-0032_animation_3.svg,lecture_02.backpropagation.key-stage-0032_animation_4.svg,lecture_02.backpropagation.key-stage-0032_animation_5.svg" class="slide-image" />

            <figcaption>
            <p    >Since the chain rule is the heart of backpropagation, and backpropagation is the heart of deep learning, we should probably take some time to see why the chain rule is true at all.<br></p><p    >If we imagine that <span>f</span> and <span>g</span> are linear functions, it’s pretty straightforward to show that this is true. They may not be, of course, but the nice thing about calculus is that locally, we can <em>treat</em> them as linear functions (if they are differentiable). In an infinitesimally small neighbourhood <span>f</span> and <span>g</span> are exactly linear.<br></p><p    >If <span>f</span> and <span>g</span> are locally linear, we can describe their behavior with a slope s and an additive constant b. The slopes,<em> </em><span>s</span><sub>f</sub> and <span>s</span><sub>g</sub>, are simply the derivatives. The additive constants we will show can be ignored. <br></p><p    >In this linear view, what the chain rule says is this: if we approximate <span>f</span>(x) as a linear function, then its slope is the slope of <span>f</span> <em>as a function of </em><em>g</em>, times the slope of <span>g</span> as a function of x. To prove that this is true, we just write down <span>f</span>(<span>g</span>(x)) as linear functions, and multiply out the brackets.<br></p><p    >Note that this doesn’t quite count as a rigorous proof, but it’s hopefully enough to give you some intuition for why the chain rule holds.<br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-030" class="anim">
            <a class="slide-link" href="test.com#slide-030" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0033_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0033_animation_0.svg,lecture_02.backpropagation.key-stage-0033_animation_1.svg" class="slide-image" />

            <figcaption>
            <p    >Since we’ll be looking at some pretty elaborate computation graphs, we’ll need to be able to deal with this situation as well: we have a computation graph, as before, but <span>f</span> depends on x through <span>two</span> <span>different </span>operations. How do we take the derivative of <span>f</span> over x?<br></p><p    >The multivariate chain rule tells us that we can simply apply the chain rule along <span>g</span>, taking <span>h</span> as a constant, and sum it with the chain rule along <span>h</span> taking <span>g</span> as a constant.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-031" class="anim">
            <a class="slide-link" href="test.com#slide-031" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0034_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0034_animation_0.svg,lecture_02.backpropagation.key-stage-0034_animation_1.svg,lecture_02.backpropagation.key-stage-0034_animation_2.svg,lecture_02.backpropagation.key-stage-0034_animation_3.svg,lecture_02.backpropagation.key-stage-0034_animation_4.svg" class="slide-image" />

            <figcaption>
            <p    >We can see why this holds in the same way as before. The short story: since all functions can be taken to be linear, their slopes distribute out into a sum<br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-032">
            <a class="slide-link" href="test.com#slide-032" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0035.svg" class="slide-image" />

            <figcaption>
            <p    >If we have more than two paths from the input to the output, we simply sum over all of them.<br></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-033">
            <a class="slide-link" href="test.com#slide-033" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0036.svg" class="slide-image" />

            <figcaption>
            <p    >With that, we are ready to show how backpropagation works. We'll start with a fairly arbitrary function to show the principle before we move on to more realistic neural networks.<br></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-034" class="anim">
            <a class="slide-link" href="test.com#slide-034" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0037_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0037_animation_0.svg,lecture_02.backpropagation.key-stage-0037_animation_1.svg,lecture_02.backpropagation.key-stage-0037_animation_2.svg,lecture_02.backpropagation.key-stage-0037_animation_3.svg" class="slide-image" />

            <figcaption>
            <p    >The first thing we do is to break up its functional form into a series of smaller operations. The entire function f is then just a chain of these small operations chained together. We can draw this in a computation graph as we did before.<br></p><p    >Normally, we wouldn’t break a function up in such small operations. This is just a simple example to illustrate the principle.<br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-035" class="anim">
            <a class="slide-link" href="test.com#slide-035" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0038_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0038_animation_0.svg,lecture_02.backpropagation.key-stage-0038_animation_1.svg,lecture_02.backpropagation.key-stage-0038_animation_2.svg,lecture_02.backpropagation.key-stage-0038_animation_3.svg" class="slide-image" />

            <figcaption>
            <p    >Now, to work out the derivative of f, we can<em> iterate the chain rule</em>. We apply it again and again, until the derivative of f over x is expressed as a long product of derivatives of operation outputs over their inputs.<br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-036">
            <a class="slide-link" href="test.com#slide-036" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0039.svg" class="slide-image" />

            <figcaption>
            <p    >We call the larger derivative of f over x the <strong>global derivative</strong>. And we call the individual factors, the derivatives of the operation output wrt to their inputs, the <span>l</span><strong>ocal derivatives</strong>.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-037">
            <a class="slide-link" href="test.com#slide-037" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0040.svg" class="slide-image" />

            <figcaption>
            <p    >This is how the backpropagation algorithm combines symbolic and numeric computation. We work out the local derivatives symbolically, and then work out the global derivative numerically.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-038" class="anim">
            <a class="slide-link" href="test.com#slide-038" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0041_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0041_animation_0.svg,lecture_02.backpropagation.key-stage-0041_animation_1.svg,lecture_02.backpropagation.key-stage-0041_animation_2.svg,lecture_02.backpropagation.key-stage-0041_animation_3.svg,lecture_02.backpropagation.key-stage-0041_animation_4.svg" class="slide-image" />

            <figcaption>
            <p    >For each local derivative, we work out the symbolic derivative with pen and paper. <br></p><p    >Note that we could fill in the <span>a</span>, <span>b</span> and <span>c</span> in the result, but we don’t. We simply leave them as is. For the symbolic part, we are only interested in the derivative of the output of each sub-operation with respect to its immediate input.<br></p><p    >The rest of thew algorithm is performed numerically.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-039" class="anim">
            <a class="slide-link" href="test.com#slide-039" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0042_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0042_animation_0.svg,lecture_02.backpropagation.key-stage-0042_animation_1.svg,lecture_02.backpropagation.key-stage-0042_animation_2.svg,lecture_02.backpropagation.key-stage-0042_animation_3.svg,lecture_02.backpropagation.key-stage-0042_animation_4.svg,lecture_02.backpropagation.key-stage-0042_animation_5.svg" class="slide-image" />

            <figcaption>
            <p    >This we are now computing things numerically, we need a specific input, in this case x = -4.499. We start by feeding this through the computation graph. For each sub-operation, we store the output value. At the end, we get the output of the function f. This is called a <strong>forward pass</strong>: a fancy term for computing the output of f for a given input.<br></p><p    >Note that at this point, we are no longer computing solutions in general. We are computing our function for a specific input. We will be computing the gradient for this specific input as well.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-040" class="anim">
            <a class="slide-link" href="test.com#slide-040" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0043_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0043_animation_0.svg,lecture_02.backpropagation.key-stage-0043_animation_1.svg,lecture_02.backpropagation.key-stage-0043_animation_2.svg,lecture_02.backpropagation.key-stage-0043_animation_3.svg" class="slide-image" />

            <figcaption>
            <p    >Keeping all intermediate values from the forward pass in memory, we go back to our symbolic expression of the derivative. Here, we fill in the intermediate values <span>a</span> <span>b</span> and <span>c</span>. After we do this, we can finish the multiplication numerically, giving us a numeric value of the gradient of f at x = -4.499. In this case, the gradient happens to be 0.<br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-041">
            <a class="slide-link" href="test.com#slide-041" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0044.svg" class="slide-image" />

            <figcaption>
            <p    >Before we try this on a neural network, here are the main things to remember about the backpropagation algorithm. <br></p><p    >Note that backpropagation by itself does not train a neural net. It just provides a gradient. When people say that they trained a network by backpropagation, that's actually shorthand for training the network by gradient descent, with the gradients worked out by backpropagation.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-042" class="anim">
            <a class="slide-link" href="test.com#slide-042" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0045_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0045_animation_0.svg,lecture_02.backpropagation.key-stage-0045_animation_1.svg,lecture_02.backpropagation.key-stage-0045_animation_2.svg" class="slide-image" />

            <figcaption>
            <p    >To explain how backpropagation works in a neural network, we extend our neural network diagram a little bit, to make it closer to the actual computation graph we’ll be using. <br></p><p    >First, we separate the hidden node into the result of the linear operation <span>k</span><sub>i</sub><span> </span>and the application of the nonlinearity <span>h</span><sub>i</sub>. Second, since we’re interested in the derivative of the loss rather than the output of the network, we extend the network with one more step:<strong> the computation of the loss</strong> (over one example to keep things simple). In this final step, the output <span>y</span> of the network is compared to the target value t from the data, producing a loss value. <br></p><p    >The loss is the function for which we want to work out the gradient, so the computation graph is the one that computes first the model output, and then the loss based on this output (and the target).<br></p><p    >Note that the model is now just a <em>subgraph</em> of the computation graph. You can think of t as another input node, like x<sub>1</sub> and x<sub>2</sub>, (but one to which the <em>model</em> doesn’t have access).</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-043" class="anim">
            <a class="slide-link" href="test.com#slide-043" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0046_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0046_animation_0.svg,lecture_02.backpropagation.key-stage-0046_animation_1.svg,lecture_02.backpropagation.key-stage-0046_animation_2.svg,lecture_02.backpropagation.key-stage-0046_animation_3.svg,lecture_02.backpropagation.key-stage-0046_animation_4.svg" class="slide-image" />

            <figcaption>
            <p    >We want to work out the gradient of the loss. This is simply the collection of the derivative of the loss over each parameter.<br></p><p    >We’ll pick two parameters, <span>v</span><sub>2</sub> in the second layer, and <span>w</span><sub>12</sub> in the first, and see how backpropagation operates. The rest of the parameters can be worked out in the same way to give us the rest of the gradient.<br></p><p    >First, we have to break the computation of the loss into operations. If we take the graph on the left to be our computation graph, then we end up with the operations of the right.<br></p><p    >To simplify things, we’ll compute the loss over only one instance. We’ll use a simple squared error loss.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-044" class="anim">
            <a class="slide-link" href="test.com#slide-044" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0047_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0047_animation_0.svg,lecture_02.backpropagation.key-stage-0047_animation_1.svg,lecture_02.backpropagation.key-stage-0047_animation_2.svg,lecture_02.backpropagation.key-stage-0047_animation_3.svg,lecture_02.backpropagation.key-stage-0047_animation_4.svg" class="slide-image" />

            <figcaption>
            <p    >For the derivative with respect to <span>v</span><sub>2</sub>, we’ll only need these two operations. Anything below doesn’t affect the result.<br></p><p    >To work out the derivative we apply the chain rule, and work out the local derivatives symbolically.<br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-045" class="anim">
            <a class="slide-link" href="test.com#slide-045" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0048_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0048_animation_0.svg,lecture_02.backpropagation.key-stage-0048_animation_1.svg,lecture_02.backpropagation.key-stage-0048_animation_2.svg,lecture_02.backpropagation.key-stage-0048_animation_3.svg" class="slide-image" />

            <figcaption>
            <p    >We then do a forward pass with some values. We get an output of 10.1, which should have been 12.1, so our loss is 4. We keep all intermediate values in memory.<br></p><p    >We then take our product of local derivatives, fill in the numeric values from the forward pass, and compute the derivative over <span>v</span><sub>2</sub>.<br></p><p    >When we apply this derivative in a gradient descent update, <span>v</span><sub>2</sub><span> </span>changes as shown below.<br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-046" class="anim">
            <a class="slide-link" href="test.com#slide-046" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0049_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0049_animation_0.svg,lecture_02.backpropagation.key-stage-0049_animation_1.svg,lecture_02.backpropagation.key-stage-0049_animation_2.svg,lecture_02.backpropagation.key-stage-0049_animation_3.svg,lecture_02.backpropagation.key-stage-0049_animation_4.svg,lecture_02.backpropagation.key-stage-0049_animation_5.svg,lecture_02.backpropagation.key-stage-0049_animation_6.svg" class="slide-image" />

            <figcaption>
            <p    >Let’s try something a bit earlier in the network: the weight <span>w</span><sub>12</sub>. We add two operations, apply the chain rule and work out the local derivatives.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-047" class="anim">
            <a class="slide-link" href="test.com#slide-047" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0050_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0050_animation_0.svg,lecture_02.backpropagation.key-stage-0050_animation_1.svg,lecture_02.backpropagation.key-stage-0050_animation_2.svg,lecture_02.backpropagation.key-stage-0050_animation_3.svg,lecture_02.backpropagation.key-stage-0050_animation_4.svg" class="slide-image" />

            <figcaption>
            <p    >Note that when we’re computing the derivative for <span>w</span><sub>12</sub>, we are also, along the way computing the derivatives for <span>y</span>, <span>h</span><sub>2</sub> and <span>k</span><sub>2</sub>.<br></p><p    >This useful when it comes to implementing backpropagation. We can walk backward down the computation graph and compute the derivative of the loss for every node. For the nodes below, we just multiply the local gradient. This means we can very efficiently compute any derivatives we need. <br></p><p    >In fact, this is <em>where the name backpropagation comes from</em>: the derivative of the loss propagates down the network in the opposite direction to the forward pass. We will show this more precisely in the last part of this lecture.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-048">
            <a class="slide-link" href="test.com#slide-048" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0051.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-049">
            <a class="slide-link" href="test.com#slide-049" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0052.svg" class="slide-image" />

            <figcaption>
            <p    >To finish up, let’s see if we can build a little intuition for what all these accumulated derivatives mean.<br></p><p    >Here is a forward pass for some weights and some inputs. Backpropagation starts with the loss, and walks down the network, figuring out at each step how every value contributed to the result of the forward pass. Every value that contributed positively to a positive loss should be lowered, every value that contributed positively to a negative loss should be increased, and so on.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-050" class="anim">
            <a class="slide-link" href="test.com#slide-050" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0053_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0053_animation_0.svg,lecture_02.backpropagation.key-stage-0053_animation_1.svg,lecture_02.backpropagation.key-stage-0053_animation_2.svg,lecture_02.backpropagation.key-stage-0053_animation_3.svg,lecture_02.backpropagation.key-stage-0053_animation_4.svg" class="slide-image" />

            <figcaption>
            <p    >We’ll start with the first value below the loss: y, the output of our model. Of course, this isn’t<em> a parameter</em> of the network, we can set it to any value we'd like. But let’s imagine for a moment that we could. What would the gradient descent update rule look like if we try to update <span>y</span>?<br></p><p    >If the output is 10, and it should have been 0, then gradient descent on y tells us to lower the output of the network. If the output is 0 and it should have been 10, GD tells us to increase the value of the output.<br></p><p    >Even though we can’t change <span>y</span> directly, this is the effect we want to achieve: we want to change the values we <em>can</em> change  so that we achieve this change in y. To figure out how to do that, we take this gradient for y, and propagate it back down the network.<br></p><p    >Note that even though these scenarios have the same loss (because of the square), the derivative of the loss has a different sign for each, so we can tell whether the output is bigger than the target or the other way around. The loss only tells us how bad we've done, but the derivative of the loss tells us where to move to make it better.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-051" class="anim">
            <a class="slide-link" href="test.com#slide-051" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0054_animation_0.svg" data-images="lecture_02.backpropagation.key-stage-0054_animation_0.svg,lecture_02.backpropagation.key-stage-0054_animation_1.svg" class="slide-image" />

            <figcaption>
            <p    >Instead of changing <span>y</span>, we have to change the values that influenced <span>y</span>.<br></p><p    >Here we see what that looks like for the <span>weights</span> of the second layer. First note that the output y in this example was <strong>too high</strong>. Since all the hidden nodes have positive values (because of the sigmoid), we end up subtracting some positive value from all the weights. This will lower the output, as expected. <br></p><p    >Second, not that the change is proportional to the input. The first hidden node <span>h</span><sub>1</sub> only contributes a factor of 0.1 (times its weight) to the value of y, so it isn't changes as much as <span>h</span><sub>3</sub>, which contributes much more to the erroneous value.<br></p><p    >Note that the current value of the weight doesn’t factor into the update. Only how much influence the weight had on the value of <span>y</span> in the forward pass. The higher the activation of the <span>source node</span>, the more the weight gets adjusted.<br></p><p    >Note also how the sign of the the derivative wrt to y is taken into account. Here, the model output was too high, so the more a weight contributed to the output, the more it gets "punished" by being lowered. If the output had been too low, the opposite would be true, and we would be adding something to the value of each weight.<br></p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-052">
            <a class="slide-link" href="test.com#slide-052" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0055.svg" class="slide-image" />

            <figcaption>
            <p    >The sigmoid activation we’ve used so far allows only positive values to emerge from the hidden layer. If we switch to an activation that also allows negative activations (like a linear activation or a <strong>tanh</strong> activation), we see that <strong>backpropagation very naturally takes the sign into account</strong>.<br></p><p    >In this case, we want to update in such a way that <span>y</span> decreases, but we note that the weight <span>v</span><sub>2</sub> is multiplied by a <em>negative</em> value. This means that (for this instance) <span>v</span><sub>2</sub> contributes<em> negatively</em> to the model output, and its value should be increased if we want to decrease the output.<br></p><p    >Note that the sign of <span>v</span><sub>2</sub> itself doesn’t matter. Whether it’s positive or negative, its value should increase.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-053">
            <a class="slide-link" href="test.com#slide-053" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0056.svg" class="slide-image" />

            <figcaption>
            <p    >We use the same principle to work our way back down the network. If we could change the output of the second node <span>h</span><sub>2</sub> directly, this is how we’d do it. <br></p><p    >Note that we now take the value of <span>v</span><sub>2</sub> to be a constant. We are working out<em> partial derivatives</em>, so when we are focusing on one parameters, all the others are taken as constant. <br></p><p    >Remember, that we want to decrease the output of the network. Since <span>v</span><sub>2</sub> makes a <em>negative</em> contribution to the loss, we can achieve this by<em> increasing</em> the activation of the source node <span>v</span><sub>2</sub> is multiplied by.<br></p><aside    >Note also that we're now using sigmoid activations again.</aside><aside    ></aside>
            </figcaption>
       </section>





       <section id="slide-054">
            <a class="slide-link" href="test.com#slide-054" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0057.svg" class="slide-image" />

            <figcaption>
            <p    >Moving down to <span>k</span><sub>2</sub>, remember that the derivative of the sigmoid is the output of the sigmoid times 1 minus that output.<br></p><p    >We see here, that in the extreme regimes, the sigmoid is <em>resistant to change</em>. The closer to 1 or 0 we get the smaller the weight update becomes.<br></p><p    >This is actually a great <em>downside</em> of the sigmoid activation, and one of the big reasons it was eventually replaced by the ReLU as the default choice for hidden units. We’ll come back to this in lecture 4.<br></p><p    >Nevertheless, this update rule tells us what the change is to <span>k</span><sub>2</sub> that we <em>want to achieve</em> by changing the gradients we can actually change (<span>the weights</span> of layer 1).</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-055">
            <a class="slide-link" href="test.com#slide-055" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0058.svg" class="slide-image" />

            <figcaption>
            <p    >Finally, we come to the weights of the first layer. As before, we want the output of the network to <em>de</em>crease. To achieve this, we want <span>h</span><sub>2</sub> to <em>in</em>crease (because <span>v</span><sub>2</sub> is negative). However, the input x<sub>1</sub> is negative, so we should decrease <span>w</span><sub>12</sub> to increase <span>h</span><sub>2</sub>. This is all beautifully captured by the chain rule: the two negatives of x<sub>1</sub> and <span>v</span><sub>2</sub> cancel out and we get a positive value which we subtract from <span>w</span><sub>12</sub>.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-056">
            <a class="slide-link" href="test.com#slide-056" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0060.svg" class="slide-image" />

            <figcaption>
            <p    >To finish up let's look at how you would implement this in code. Here is the forward pass: computing the model output and the loss, given the inputs and the target value.<br></p><p    >Assume that <span>k</span> and <span>y</span> are initialized with 0s or random values. We'll talk about initialization strategies in the 4th lecture.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-057">
            <a class="slide-link" href="test.com#slide-057" title="Link to this slide.">link here</a>
            <img src="lecture_02.backpropagation.key-stage-0061.svg" class="slide-image" />

            <figcaption>
            <p    >And here is the backward pass. We compute gradients for every node in the network, regardless of whether the node represents a parameter. When we do the gradient descent update, we'll use the gradients of the parameters, and ignore the rest.<br></p><p    >Note that we don’t implement the derivations from slide 44 directly. Instead, we work backwards down the neural network: computing the derivative of each node as we go, by taking the derivative of the loss over the outputs and multiplying it by the local derivative.</p><p    ></p>
            </figcaption>
       </section>


</article>
