import mate
from ..trainers.trainers.classification_trainer import TrainerModule
from ..models.googlenet import GoogleNet
from ..data.loaders.cifar10 import get_data
from flax import linen as nn
import jax

train_loader, val_loader, test_loader = get_data(
    dataset_path="data", train_batch_size=64, test_batch_size=128
)
model = GoogleNet(num_classes=10, act_fn=nn.relu)

if mate.is_train:
    val_result = TrainerModule.train_classifier(
        model=model,
        optimizer_name="adamw",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        exmp_imgs=jax.device_put(next(iter(train_loader))[0]),
        num_epochs=2,
        checkpoint_path=mate.checkpoint_path,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=mate.results_folder,
    )
    mate.result({"val_acc": val_result})
elif mate.is_test:
    test_result = TrainerModule.test_classifier(
        model=model,
        optimizer_name="adamw",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        exmp_imgs=jax.device_put(next(iter(train_loader))[0]),
        checkpoint_path=mate.checkpoint_path,
        test_loader=test_loader,
        save_path=mate.results_folder,
    )
    mate.result({"test_acc": test_result})
