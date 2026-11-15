"""Measured update ETA; epoch work is measured separately, never silently counted as updates."""
import time
from tqdm.auto import tqdm


class TrainingProgress:
    def __init__(self, total_steps, epochs, monitor, initial_step=0, eval_every=50,
                 initial_epoch_eval_seconds=0.):
        self.initial_step = initial_step
        self.eval_every = eval_every
        self.checkpoint_seconds = 0.
        self.checkpoint_count = 0
        self.initial_epoch_eval_seconds = initial_epoch_eval_seconds
        self.started = time.monotonic()
        self.update_seconds = 0.
        self.eval_seconds = 0.
        self.completed_epochs = 0
        self.epochs = epochs
        self.monitor = monitor
        self.bar = tqdm(total=total_steps, initial=initial_step, desc='训练更新', unit='step',
                        dynamic_ncols=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]')

    def step(self, seconds, loss, epoch):
        self.update_seconds += seconds
        self.bar.update()
        remaining_updates = (self.bar.total - self.bar.n) * self.update_seconds / max(1,self.bar.n-self.initial_step)
        remaining_epochs = self.epochs - self.completed_epochs
        # Initial full evaluation is a useful estimate before the first epoch ends.
        remaining_eval = (self.eval_seconds / self.completed_epochs * remaining_epochs
                          if self.completed_epochs else self.initial_epoch_eval_seconds * remaining_epochs)
        if self.checkpoint_count:
            remaining_eval += self.checkpoint_seconds/self.checkpoint_count * (self.bar.total//self.eval_every-self.bar.n//self.eval_every)
        values = dict(elapsed_seconds=time.monotonic() - self.started,
                      update_seconds=self.update_seconds, evaluation_seconds=self.eval_seconds,
                      eta_updates_seconds=remaining_updates,
                      eta_total_seconds=remaining_updates + remaining_eval,
                      eta_includes_evaluation=int(self.completed_epochs > 0 or self.checkpoint_count > 0
                                                  or self.initial_epoch_eval_seconds > 0),
                      completed_steps=self.bar.n, total_steps=self.bar.total)
        self.bar.set_postfix(epoch=epoch, loss=f'{loss:.3f}',
                             ETA=f'{remaining_updates + remaining_eval:.0f}s', refresh=False)
        if self.bar.n == 1 or self.bar.n % 10 == 0 or self.bar.n == self.bar.total:
            self.publish(values)

    def publish(self, values):
        for name, value in values.items():
            self.monitor.writer.add_scalar(f'time/{name}', value, self.bar.n)
        self.monitor.writer.flush()

    def evaluation(self, seconds):
        self.eval_seconds += seconds
        self.completed_epochs += 1
        self.publish(dict(elapsed_seconds=time.monotonic() - self.started,
                          evaluation_seconds=self.eval_seconds,
                          last_epoch_evaluation_seconds=seconds))

    def checkpoint(self, seconds):
        self.checkpoint_seconds += seconds
        self.checkpoint_count += 1
        self.publish(dict(periodic_evaluation_and_save_seconds=self.checkpoint_seconds))

    def close(self, completed=True):
        values = dict(elapsed_seconds=time.monotonic() - self.started)
        if completed: values.update(eta_updates_seconds=0, eta_total_seconds=0)
        self.publish(values)
        self.bar.close()
