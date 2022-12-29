import mate
from ..trainers.classification_trainer import TrainerModule
from ..models.resnet import ResNet
from ..data_loaders.cifar10 import get_data
from flax import linen as nn
import jax

train_loader, val_loader, test_loader = get_data(
    dataset_path="data", train_batch_size=8, test_batch_size=32
)
model = ResNet(num_classes=10, act_fn=nn.relu)

if mate.is_train:
    val_result = TrainerModule.train_classifier(
        model=model,
        save_path=mate.save_dir,
        optimizer_name="adamw",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        exmp_imgs=jax.device_put(next(iter(train_loader))[0]),
        num_epochs=2,
        checkpoint_path=mate.checkpoint_path,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    mate.result({"val_acc": val_result})
elif mate.is_test:
    test_result = TrainerModule.test_classifier(
        model=model,
        save_path=mate.save_dir,
        optimizer_name="adamw",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
        exmp_imgs=jax.device_put(next(iter(train_loader))[0]),
        checkpoint_path=mate.checkpoint_path,
        test_loader=test_loader,
    )
    mate.result({"test_acc": test_result})
