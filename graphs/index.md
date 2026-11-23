---
title: "Graphs"
slides: true
---
<nav class="menu">
    <ul>
        <li class="home"><a href="/">Home</a></li>
        <li class="name">Graphs</li>
                <li><a href="#video-000">Why convolutions?</a></li>
                <li><a href="#slide-013">Learning convolutions</a></li>
                <li><a href="#slide-036">Convolutional architectures</a></li>
                <li><a href="#slide-050">Convolutional architectures</a></li>
        <li class="pdf"><a href="https://dlvu.github.io/pdfs/lecture08.graphs.annotated.pdf">PDF</a></li>
    </ul>
</nav>

<article class="slides">



       <section id="slide-001">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-001" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0001.svg" class="slide-image" />

            <figcaption>
            <p    ><br></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-002" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-002" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0002anim0.svg" data-images="lecture07.graphs.key-stage-0002anim0.svg,lecture07.graphs.key-stage-0002anim1.svg" class="slide-image" />

            <figcaption>
            <p    >This is a graph. A collection of dots joined by lines. The dots are called <strong>nodes</strong> or vertices and the lines are called <strong>links </strong>or edges. <br></p><aside    >We’ll try to stick with nodes and links. <br></aside><p    >Graphs are very useful, because they are a very versatile way of <em>representing </em>many things. In the context of this lecture, we will focus on <strong>graphs as data</strong>. That usually means that the nodes in the graph represent entities in the real world, and the links represent relations between them. <br></p><p    >In this example, the nodes represent people, and the links express who follows whom on social media. </p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-003">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-003" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0003.svg" class="slide-image" />

            <figcaption>
            <p    >If the links have a direction we call the graph <em>directed</em>. Examples of directed relations are: following someone on social media, owing somebody money, being someone’s parent and so on. If you reverse the arrow, the meaning of the link changes. <br></p><p    >If the links don’t change meaning when you reverse them, we call the graph undirected. Examples of undirected relations are: two cities being connected by rail, people being friends, people having met, and so on. <br></p><p    >If an edge connects a link to itself, we call that a self-loop. Self-loops are sometimes impossible, like being your own parent, or always true, like being a your own family member. If they are always true that are usually left out. There are however cases in which a self-loop is meaningful (that is, it is sometimes true), like countries that are their own capital (e.g. Vatican city).<br></p><p    >If there can be two edges between nodes, for example to express different loans people have agreed on, we call that a multi-edge, and if this is allowed, we call the graph a multi-graph. <br></p><p    >A graph that doesn’t have multi-edges or self-loops is called a  <strong>simple graph </strong>(it can be directed or undirected). <br></p><aside    >There is no name for a graph that has self-loops but no multi-edges. This does happen often, but in thos cases, you just have to spell it out. </aside><aside    ></aside>
            </figcaption>
       </section>





       <section id="slide-004">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-004" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0004.svg" class="slide-image" />

            <figcaption>
            <p    >One particularly useful type of graph in the representation of data is a <strong>knowledge graph</strong>. In a knowledge graph, we have edges of different types, so that we can express different relations between entities. You can think of a knowledge graph as several graphs expressing different relations joined together. <br></p><p    >You could do this with various types of graphs, but the most common rules for knowledge graphs are:<br></p><p     class="list-item">All links are directed.<br></p><p     class="list-item">Self-loops are allowed<br></p><p     class="list-item">Multi-edges are allowed but only of different relations. A single relation can only connect entities in one direction once.<br></p><p    >We’ll put a special focus on knowledge graphs in this lecture, since they are so useful as representations of datasets.<br></p><p    ><br><br></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-005" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-005" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0005anim0.svg" data-images="lecture07.graphs.key-stage-0005anim0.svg,lecture07.graphs.key-stage-0005anim1.svg" class="slide-image" />

            <figcaption>
            <p    >One standard way of representing a knowledge graph is the <strong>Resource Description Framework (RDF)</strong>. This uses URLs for the entities and relations which allows different datasets to refer to the same entities in a coherent way.<br></p><aside    >They’re actually called IRIs (internationalized resource identifiers). This is to emphasize that they’re <span>names</span>, rather than <span>addresses</span>. Something usually happens when you put them in the address bar of a browser, so they’re URLs as well, but that is not their primary purpose. <br></aside><p    >RDF also allows “literal nodes” which can be used to encode non graph data like numbers, dates, natural language or images as special nodes in the graph. This makes RDF a particularly useful format for integrating different types of multi-modal data. </p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-006">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-006" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0006.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-007">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-007" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0007.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-008">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-008" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0008.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-009" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-009" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0009anim0.svg" data-images="lecture07.graphs.key-stage-0009anim0.svg,lecture07.graphs.key-stage-0009anim1.svg" class="slide-image" />

            <figcaption>
            <p    >The aim of link prediction is to predict links that are missing from the graph. This could be because the graph is genuinely incomplete, but it could also be a good way to predict links that <em>should</em> be in the graph for some reason. <br></p><p    >For example, if we train a model that thinks that Claire is likely to follow Don, then that would be a good recommendation for the social network to make to Claire.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-010" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-010" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0010anim0.svg" data-images="lecture07.graphs.key-stage-0010anim0.svg,lecture07.graphs.key-stage-0010anim1.svg" class="slide-image" />

            <figcaption>
            <p    >Link prediction in knowledge graphs is especially powerful. It’s like training a whole collection on classifiers and recommender systems in one go. <br></p><p    >If we have just two types of nodes (say users and movies) and one relation (indicating whether the user liked the movie) then link prediction becomes movie recommendation. By expanding the knowledge graph we can include information about the users and about the movies. <br></p><aside    >The connection between recommender systems and link prediction is a very fertile idea to keep in mind. For more on this, see the recommender systems lecture in our BSc course machine learning here: <a href="https://mlvu.github.io/embeddingmodels/"><strong>https://mlvu.github.io/embeddingmodels/</strong></a> (there’s also a connection to PCA there).</aside><aside    ></aside>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-011">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-011" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0011.svg" class="slide-image" />

            <figcaption>
            <p    ><strong>Node classification</strong> is another task: for each node, we are given a label, which we should try to predict.<br></p><p    >If we have vector representations for our nodes, we can use those in a regular classifier, but the question is, how do we get those embeddings, and how do we ensure that they capture the required information?<br></p><p    >We can’t just assign node embeddings like before, and apply gradient descent on the classification loss. That would ignore the graph structure entirely and would train each embedding in isolation to produce a particular clasification. We wouldn’t generalize <em>between</em> nodes.<br></p><aside    ><br></aside><aside    ></aside>
            </figcaption>
       </section>





       <section id="slide-012">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-012" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0012.svg" class="slide-image" />

            <figcaption>
            <p    >Finally, you may have a labeling task where the label applies not to one node but to a whole graph. One example is the MUTAG dataset, where the task is to predict whether a particular molecule, represented as a graph, is mutagenic. <br></p><p    >You  can of course also do graph<em> regression</em>, where the task is to assign a number to each graph in the dataset. <br></p><aside    >The atoms in MUTAG have types. These aren’t drawn, but they are usually included in the task as node features. <br></aside><aside    >Note that in node classification your dataset is one (large) graph, whereas in graph classification it consists of many (small) graphs. There is a symmetry here with what we said for sequences: there are datasets that consist of a single sequence, or datasets that are a set of sequences, each with a label. </aside><aside    ></aside>
            </figcaption>
       </section>





       <section id="slide-013">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-013" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0013.svg" class="slide-image" />

            <figcaption>
            <p    ><br></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-014" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-014" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0014anim0.svg" data-images="lecture07.graphs.key-stage-0014anim0.svg,lecture07.graphs.key-stage-0014anim1.svg" class="slide-image" />

            <figcaption>
            <p    >We’ll start with link prediction.Assume we have a graph with N nodes. Our job is to generate N vectors representing these nodes. Or if we concatenate them, a matrix with N columns. <br></p><p    >The dimensionality of the feature vectors, k, is a hyperparameter that we (usually) set ourselves. Usually setting this too high leads to overfitting and setting it too low leads to underfitting.<br></p><p    >Link prediction can be either a task in itself—if you really want to predict new links that might be true—of a proxy task: something you don’t really want to do, but that will get you high quality node embeddings that you can then use later for something else.<br></p><p    >This is a bit like learning word embeddings with Word2Vec. The prediction task itself may not be what we’re interested in, but it’s a task that allows the model to learn good representations, and a task that doesn’t require us to do any labelling.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-015">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-015" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0015.svg" class="slide-image" />

            <figcaption>
            <p    >The key idea of link prediction is to come up with node representations that can predict, for two given nodes, whether there is or should be a link between them. For example, if your graph is a friendship  network, you could predict whether these two people should be friends.<br></p><p    >We then train our node representations to make this prediction as accurate as possible. <br></p><p    >You can think of this as a <em>surrogate task</em>.Sometimes we are actually interested in predicting these links, but even if we aren’t, the resulting node representations can be used in other tasks, like node labelling.<br></p><aside    >We’re focusing here on link prediction methods that work by building embedding vectors for the nodes in the graph. There may be other ways to solve the problem, but this is the most popular family of approaches. </aside><aside    ></aside>
            </figcaption>
       </section>





       <section id="slide-016" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-016" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0017anim0.svg" data-images="lecture07.graphs.key-stage-0017anim0.svg,lecture07.graphs.key-stage-0017anim1.svg" class="slide-image" />

            <figcaption>
            <p    >To figure out how we might do this, imagine first that we had the resources to manually set features for each node in our graph. We’ll stick with our example of a friendship graph. What features might tell us whether two people should be friends? Here we’ve gone with how physically active they are, how introverted they are, and politically conservative they are. We’ll assume (perhaps naively) that if these match between two people then they are more likely to make good friends.<br></p><p    >Each feature is a number between -inf and +inf. <br></p><p    >What this shows, is that in this situation, where we have known features, the dot product is a good score to help us predict whether two people are ()or should be) friends</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-017">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-017" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0018.svg" class="slide-image" />

            <figcaption>
            <p    >If we don’t have features, and instead want to learn the values of the embeddings, what we can do is take the dot product and train it to be high for pairs of connected people (friend) and low for pairs of unconnected people (non friends). If A is our adjacency matrix, we can use squared error for our loss and train the dot product to mimic the values in the adjacency matrix.<br></p><p    >This objective is something we can minimize by gradient descent. <br></p><p    >What’s wrong with this idea?<br></p><p     class="list-item">dot product needs to mimic value between 0 and 1<br></p><p     class="list-item">assumes all values are known</p><p     class="list-item"></p>
            </figcaption>
       </section>





       <section id="slide-018" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-018" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0019anim0.svg" data-images="lecture07.graphs.key-stage-0019anim0.svg,lecture07.graphs.key-stage-0019anim1.svg,lecture07.graphs.key-stage-0019anim2.svg" class="slide-image" />

            <figcaption>
            <p    >One common way to get scores that range from 0 to 1 is to model them as probabilities. We take the dot product of the two node embeddings and we pass it through a sigmoid function. We then<em> interpret</em> this as the probability that node <strong>a</strong> links to node <strong>b</strong>. <br></p><p    >Once we have that, we can compute the log likelihood of the data under our model parameters (all the embedding vectors) and use that as our loss. </p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-019">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-019" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0020.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-020">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-020" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0021.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-021">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-021" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0022.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-022">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-022" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0023.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-023">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-023" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0024.svg" class="slide-image" />

            <figcaption>
            <p    >Example of a (simple) query. We’ll be looking at more elaborate queries later in the course.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-024" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-024" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0025anim0.svg" data-images="lecture07.graphs.key-stage-0025anim0.svg,lecture07.graphs.key-stage-0025anim1.svg,lecture07.graphs.key-stage-0025anim2.svg,lecture07.graphs.key-stage-0025anim3.svg,lecture07.graphs.key-stage-0025anim4.svg,lecture07.graphs.key-stage-0025anim5.svg,lecture07.graphs.key-stage-0025anim6.svg,lecture07.graphs.key-stage-0025anim7.svg" class="slide-image" />

            <figcaption>
            <p    >More on the evaluation part later. </p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-025" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-025" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0026anim0.svg" data-images="lecture07.graphs.key-stage-0026anim0.svg,lecture07.graphs.key-stage-0026anim1.svg,lecture07.graphs.key-stage-0026anim2.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-026">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-026" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0027.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-027">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-027" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0028.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-028">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-028" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0029.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-029">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-029" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0030.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-030">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-030" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0031.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-031">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-031" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0032.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-032" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-032" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0033anim0.svg" data-images="lecture07.graphs.key-stage-0033anim0.svg,lecture07.graphs.key-stage-0033anim1.svg" class="slide-image" />

            <figcaption>
            <p    >Where log loss takes inspiration from logistic regression, hinge loss takes inspiration from SVMs and the max-margin loss. In practice, it boils down to passing the raw score x through a hinge function max(x + m, 0) for negatives and - max(x +  m, 0) for positives.<br></p><p    >We then take the sign of the score as the prediction (a positive score is a positive triple).<br></p><p    >This means that positives that get a very high score don’t count in the loss function. Only the ones that the model gets wrong (true triples with a negative score) or just right (true triples with a low positive score) do anything to the loss function. The mechanism is similar to the SVM in that only a few key examples end up determining the parameters of the model.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-033">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-033" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0034.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-034">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-034" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0035.svg" class="slide-image" />

            <figcaption>
            <p    >The pointwise hinge loss is shown, but you can do the same thing with a log loss.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-035">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-035" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0036.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-036">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-036" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0037.svg" class="slide-image" />

            <figcaption>
            <p    ></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-037">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-037" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0038.svg" class="slide-image" />

            <figcaption>
            <p    >Think back to our movie recommendation example. Say we predict from user. movie embeddings by a simple dot product (the higher, the more likeley the user is to like the movie).<br></p><p    >In that case, all the predictions we can make (i.e. all the dot product) for a matrix that we can calculate by multiplying the matrix of user embeddings by the matrix of movie embeddings as shown.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-038">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-038" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0039.svg" class="slide-image" />

            <figcaption>
            <p    >It is important to carefully consider our validation protocol. In other words: how do we withhold test data to train on.<br></p><p    >Let’s start with recommender systems. At first, you might think that it’s a good idea to just withhold some users.<br></p><p    >However, this doesn’t work: if we don’t see the users during training, we won’t learn embeddings for them, which means we can’t generate predictions.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-039">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-039" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0040.svg" class="slide-image" />

            <figcaption>
            <p    >How about if we withhold some movies? The same thing happens.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-040">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-040" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0041.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-041" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-041" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0042anim0.svg" data-images="lecture07.graphs.key-stage-0042anim0.svg,lecture07.graphs.key-stage-0042anim1.svg" class="slide-image" />

            <figcaption>
            <p    >This is related to the difference between <strong>inductive</strong> and <strong>transductive </strong>learning. In the transductive setting, the learning is allowed to see the features of all data, but the labels of only the training data.<br></p><p    >In short, when you have embedding vectors for entities in your data, you can only do transductive learning. The reason is simple: you cannot learn what to do for entities you’ve never seen before, because you won’t have trained their embeddings yet. </p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-042">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-042" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0043.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-043">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-043" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0044.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-044">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-044" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0045.svg" class="slide-image" />

            <figcaption>
            <p    >In the case of node classification, we provide the algorithm with the whole graph, and a table linking the node ids to the labels. In this table, we withhold some of the labels. The model is however, aware of the nodes for which we’ve withheld the labels.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-045">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-045" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0046.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-046" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-046" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0047anim0.svg" data-images="lecture07.graphs.key-stage-0047anim0.svg,lecture07.graphs.key-stage-0047anim1.svg,lecture07.graphs.key-stage-0047anim2.svg,lecture07.graphs.key-stage-0047anim3.svg,lecture07.graphs.key-stage-0047anim4.svg,lecture07.graphs.key-stage-0047anim5.svg,lecture07.graphs.key-stage-0047anim6.svg,lecture07.graphs.key-stage-0047anim7.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-047">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-047" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0048.svg" class="slide-image" />

            <figcaption>
            <p    >Accuracy is rarely used, since most models get such low accuracy that we don’t get a very good signal of how well the models are doing relative to each other. However, it can help you understand what the other two metrics mean.<br></p><p    >The diagram illustrates each metric for an imagined test set of 6 triples (checking only the tail corruption). We test the model and for each triple, find what the rank of the true triple is. This is given on the horizontal axis. The metric is then computed by taking the average length of the bars (with the tallest ones having length 1).<br></p><p    >Note that this means that all metrics are values between 0 and 1. We can also conclude that the accuracy is always a lowerbound for the hits@k and for the mrr.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-048">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-048" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0049.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-049">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-049" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0050.svg" class="slide-image" />

            <figcaption>
            <p    >This paper is a good source for a detailed explanation of the knowledge graph link prediction protocol. </p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-050">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-050" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0051.svg" class="slide-image" />

            <figcaption>
            <p    ></p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-051">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-051" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0052.svg" class="slide-image" />

            <figcaption>
            <p    >In the previous section we saw how to do link prediction with node embeddings. This is an interesting task in itself, but it can also be a good proxy task to train reasonable <em>node representations</em> that we can then use in a task like node classification. <br></p><p    >In this section we’ll go a bit further, and focus on node classification by deep neural network. </p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-052">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-052" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0053.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-053">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-053" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0054.svg" class="slide-image" />

            <figcaption>
            <p    >By analogy with the sequence-to-sequence layer, we can design <strong>a graph-to-graph layer</strong>. We have a graph, with a certain number of nodes. Each node is represented by a vector. This could be an embedding vector, a feature vector, or the output of some process below this layer. <br></p><p    >The layer operates on the graph, and produces new vectors representing the nodes. Ideally, like the (non-trivial) sequence-to-sequence layers propagate information over the time dimension, the graph-to-graph layer propagates information over the graph structure. </p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-054">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-054" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0055.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-055">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-055" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0056.svg" class="slide-image" />

            <figcaption>
            <p    >Here is the very simplest graph neural network we can build. The simple graph convolution. We start with node embeddings. Then the aggregation layer replaces each node embedding with the sum over the embeddings of all the neighbors of that node. Then we apply a simple linear layer, and a non-linearity.  </p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-056">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-056" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0057.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-057">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-057" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0058.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-058">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-058" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0059.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-059">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-059" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0060.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-060">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-060" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0061.svg" class="slide-image" />

            <figcaption>
            <p    >Remember that we should be on the lookout for activations blowing up. If we sum together the vectors for a node with five neighbors, then the output activations will be five times as big as the input activations. <br></p><p    >To fix this, we can simply replace our sum layer with an average.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-061" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-061" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0062anim0.svg" data-images="lecture07.graphs.key-stage-0062anim0.svg,lecture07.graphs.key-stage-0062anim1.svg,lecture07.graphs.key-stage-0062anim2.svg" class="slide-image" />

            <figcaption>
            <p    >To write this in a matrix view, we create a degree matrix <strong>D</strong>. This is just a diagonal matrix with the degrees of the nodes along the diagonal. The inverse of a diagonal matrix just has 1/d where the original matrix has d, so <strong>D</strong><sup>-1</sup> gives us all the values we want to multiply by along the diagonal.<br></p><p    >If we multiply <strong>D</strong><sup>-1 </sup>by <strong>A </strong>then we then we are row normalizing the adhjacency matrix. That is, all the values in one row sum to one. After row-normalizing, multiplying by the adjacency matrix computes precisely the average we are looking for. </p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-062" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-062" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0063anim0.svg" data-images="lecture07.graphs.key-stage-0063anim0.svg,lecture07.graphs.key-stage-0063anim1.svg,lecture07.graphs.key-stage-0063anim2.svg,lecture07.graphs.key-stage-0063anim3.svg,lecture07.graphs.key-stage-0063anim4.svg,lecture07.graphs.key-stage-0063anim5.png,lecture07.graphs.key-stage-0063anim6.png,lecture07.graphs.key-stage-0063anim7.png,lecture07.graphs.key-stage-0063anim8.png" class="slide-image" />

            <figcaption>
            <p    >Of course, we don’t just care about keeping the activations in check. <br></p><p    >We know from previous derivations that if Y is computed by the matrix multiplication AX, then the gradients for X are computed by mutiplying the gradients  for Y by the transpose of A. <br></p><p    >This tells us that if we row-normalize, as we did in the last slide, we can end up shrinking and blowing up the gradients. In the example, we are multiplying the incoming gradient vectors for node 2 by 1/2 and 1/5, which sums to less than one. For node 3, the weights sum to 4!<br></p><p    >We can fix this by column-normalizing instead.<br></p><aside    >We give the gradient derivation here to make things self-contained, but it’s the same as we saw in the backpropagation lecture. </aside><aside    ></aside>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-063" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-063" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0064anim0.svg" data-images="lecture07.graphs.key-stage-0064anim0.svg,lecture07.graphs.key-stage-0064anim1.svg" class="slide-image" />

            <figcaption>
            <p    >We achieve column normalization simply by right multiplying the inverse degree matrix. <br></p><p    >With column normalization, the gradients propagate perfectly, but of course now we have a problem with the activations: they can now shrink and blow up as before. <br></p><p    >For directed graphs, we often just row-normalize and call it a day. If the network isn’t too deep, the gradients should still propagate. <br></p><p    >For undirected graphs however, it’s more common to do a kind of mixture between row and column normalization.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-064">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-064" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0065.svg" class="slide-image" />

            <figcaption>
            <p    >The trick is to break <strong>D</strong><sup>-1</sup> up into two matrices that give you if you multiply them together. This is easy to do with diagonal matrices, since multiply them is just element-wise multiplying the diagonal elements. That means that if we take the square roots of the diagonal elements (<strong>D</strong><sup>-1/2</sup>), we get <strong>D</strong><sup>-1</sup> back.<br></p><p    >We can now row normalize “a bit” and column normalize “a bit” by premultlying by the square root degree matrix <em>and</em> post-multiplying by it.<br></p><p    >This gives us a bit of control over the activations blowing up and over the gradients blowing up. The weights are not fully normalized in the row or column direction (you can check that they never sum to one) but their sums should never sum a value that is too far below or above 1.</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-065">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-065" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0066.svg" class="slide-image" />

            <figcaption>
            <p    >Which operations can we use for aggregation?</p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-066" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-066" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0067anim0.svg" data-images="lecture07.graphs.key-stage-0067anim0.svg,lecture07.graphs.key-stage-0067anim1.svg,lecture07.graphs.key-stage-0067anim2.svg" class="slide-image" />

            <figcaption>
            <p    >We can think of the aggregation step as taking the current node representation, and mixing in some of the representation of the neighbors. We usually don’t want to throw the current representation of the current node away, however, we just want to add to it. <br></p><p    >To achieve that, we should include the current node x among its own neighbors N(x). Or, equivalently, we should assume that each neighbor has a self-loop. <br></p><p    >Often, the self loops are given their own separate weight matrix <strong>V</strong>. This allows the network to treat the information propagating over the self loops differently from the information coming in from the neighbors.<br></p><p    >You can think of this as applying two GCNs: one for the regular graph, and one for the self loops.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-067" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-067" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0068anim0.svg" data-images="lecture07.graphs.key-stage-0068anim0.svg,lecture07.graphs.key-stage-0068anim1.svg,lecture07.graphs.key-stage-0068anim2.svg,lecture07.graphs.key-stage-0068anim3.svg,lecture07.graphs.key-stage-0068anim4.svg,lecture07.graphs.key-stage-0068anim5.svg,lecture07.graphs.key-stage-0068anim6.svg,lecture07.graphs.key-stage-0068anim7.svg,lecture07.graphs.key-stage-0068anim8.svg" class="slide-image" />

            <figcaption>
            <p    >Here’s what a 2-layer node classification GCN looks like step by step, with the self loops in place.</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-068" class="anim">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-068" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0069anim0.svg" data-images="lecture07.graphs.key-stage-0069anim0.svg,lecture07.graphs.key-stage-0069anim1.svg,lecture07.graphs.key-stage-0069anim2.svg" class="slide-image" />

            <figcaption>
            <p    >We can extend this idea to allow us to handle knowledge graphs. We simply apply a separate GCN for each relation, each with its own weight matrix. and then sum together their outputs. before applying the non-linearity.<br></p><aside    >We really should be averaging over the  relations rather than summing, to stop the gradientsd from blowing up, but for the shallow RGCNs we see in practice, it doesn’t seem to matter. The summed version is the canonical formula.<br></aside><p    >We can now treat the self loops as just another relation (the “identity” relation).<br></p><p    >We also normally include all inverse relations: if there is an arrow from a to b with relation r, we add an arrow from b to a with relation r<sup>-1</sup>. <br></p><aside    >In the image we’ve given these intuitive names, but for the model we don’t need to know what these are. We just add one inverse relation for every relation in the given graph. <br></aside><p    >The reason for the inverse links is that in a directed GCN, the model passes information along the direction of the node. However this direction doesn’t usually mean that information only flows in one direction. It just indicates which node is the object and which node is the subject. We could let information flow in two directions, but that would be like making the graph undirected. Ie.e removing information. By creating inverse relations, information flows in two directions, but the information of the edge directions is retained. <br></p><p    >In short, if we are given a knowledge graph with R relations, our RGCN has 2R + 1 relations (we don’t add an inverse link for the identity).</p><p    ></p>
            </figcaption>
            <span class="hint">click image for animation</span>
       </section>





       <section id="slide-069">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-069" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0070.svg" class="slide-image" />

            <figcaption>
            <p    >RGCNs with many relations are prone to overfitting, because they have a large number of weights (the double descent phenomenon has not been observed yet in GCNs). <br></p><p    >One way of regularizing them is to make the weigh matrices not free weights, but linear combinations of a small number of basis matrices. The picture shows the idea for two basis matrices V<sub>1</sub> and V<sub>2</sub>. </p><p    ></p>
            </figcaption>
       </section>





       <section id="slide-070">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-070" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0071.png" class="slide-image" />

            <figcaption>
            <p    >We can define a whole family of GNNs by alternating two types of layers. One that propagates information over the graph, and one that computes locally on the individual node embeddings. These are called <strong>aggregate</strong> and <strong>update</strong>. So long as we make sure that aggregate is permutation equivariant and update is local (i.e. pointwise), we can pick any function we like. <br></p><p    >This is called the <strong>message passing framework</strong>.  The idea is that the neural network computes its function by passing messages aling the links of the network. <br></p><aside    >The “update” is the FF layer in the earlier slide. In this case, the self-connection is included as an input there, but that might be different in other definitions of message passing.</aside><aside    ></aside>
            </figcaption>
       </section>





       <section id="slide-071">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-071" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0072.png" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>





       <section id="slide-072">
            <a class="slide-link" href="https://dlvu.github.io/graphs#slide-072" title="Link to this slide.">link here</a>
            <img src="lecture07.graphs.key-stage-0073.svg" class="slide-image" />

            <figcaption>
            <p    ></p>
            </figcaption>
       </section>


</article>
