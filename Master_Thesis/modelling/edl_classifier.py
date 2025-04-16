import torch
from torch import nn
import torch.nn.functional as F

from modelling.model_utils import get_smp_model


class EDLClassifier(nn.Module):
    """
    Evidential Deep Learning Classifier
    """
    def __init__(self, num_classes, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        # Get the smp model
        self.model = get_smp_model(num_classes=num_classes, encoder=encoder, decoder=decoder)

    def forward(self, x):
        """
        instead of using regular softmax or sigmoid to output a probability distribution over the classes,
        we output a positive vector, using a softplus on the logits, as the evidence over the classes.
        softplus smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.
        
        Using the **softplus function** itself does not directly create a Dirichlet distribution. However, it can be part of a process that parameterizes a Dirichlet distribution. Here's the breakdown:

        ### 1. **Dirichlet Distribution Basics**:
        - A Dirichlet distribution is parameterized by a set of concentration parameters,
        \( \boldsymbol{\alpha} \) = [\α_1, \α_2, ..., α_K] , where α_k > 0 for each class k.
        - The \( \boldsymbol{\alpha} \) values control the shape of the distribution.

        ### 2. **Role of Softplus**:
        The **softplus function**: softplus(x) = log(1 + e^x)
        is commonly used to ensure positivity, as it maps any real-valued input x to a positive output. This makes it
        suitable for constructing parameters that require positive values, such as α_k in a Dirichlet distribution.

        ### 3. **How Softplus Can Be Used**:
        To create a Dirichlet distribution:
        - A neural network might output logits (real-valued numbers) as intermediate parameters.
        - These logits are then passed through the **softplus function** to ensure positivity, converting them into
        valid concentration parameters α_k.
        - The resulting \( \boldsymbol{\alpha} \) values can then define a Dirichlet distribution.

        ### 4. **Key Insight**:
        The softplus function itself does not "create" a Dirichlet distribution; it transforms raw outputs (e.g., logits)
        into a valid space for \( \boldsymbol{\alpha} \). The Dirichlet distribution is formed when these 
        α_k parameters are used as inputs to the Dirichlet probability density function.

        ### 5. **Why Softplus Is Commonly Used**:
        - **Smoothness**: Softplus is a smooth approximation of the ReLU function, which is often used for
        positive constraints.
        - **Avoids Zero Values**: Unlike ReLU, which can output zero, softplus ensures strictly positive values,
        which is critical for α_k(since α_k = 0 is undefined in a Dirichlet distribution).

        ---

        ### Example in Practice:
        If you're using softplus in a model for uncertainty estimation or Bayesian neural networks, it's likely part of a pipeline like this:
        1. Neural network outputs raw logits: z_k.
        2. Apply softplus: α_k = softplus(z_k).
        3. Use \( \boldsymbol{\alpha} \) = [\α_1, \α_2, ..., α_K]  as parameters for a Dirichlet distribution.

        This process allows the model to generate and update a Dirichlet distribution as part of its predictions.
        """
        logits = self.model(x)  # torch.Size([4, 13, 1024, 1024])
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="nearest")
        return F.softplus(logits)  # SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive. Softplus(x)= 1/β∗log(1+exp(β∗x))


    @torch.inference_mode()
    def predict(self, x, return_uncertainty=True):
        # Calls forward function
        evidences = self(x)
        # alphas are the parameters of the Dirichlet distribution that models the probability distribution over the class probabilities and strength is the Dirichlet strength
        alphas = evidences + 1.0
        strength = torch.sum(alphas, dim=1, keepdim=True)
        probabilities = alphas / strength

        if return_uncertainty:
            total_uncertainty = self.num_classes / strength
            beliefs = evidences / strength
            return probabilities, total_uncertainty, beliefs
        else:
            return probabilities
