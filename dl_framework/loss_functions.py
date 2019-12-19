from torch import nn


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, base_loss, layer_ids, layer_wgts):
        """"
        m_feat: enthält das vortrainierte Netz
        loss_features: dort werden alle features gespeichert, deren Loss
        man berechnen will
        """
        super().__init__()
        self.m_feat = m_feat
        self.base_loss = base_loss
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        """"
        Hier werden Kopien der gespeicherten Aktivierungsfunktionen
        abgegriffen und Kopien davon gespeichert. Sowohl einmal für
        die Wahrheit "target" und einmal für die Prediction "input"
        aus dem Generator.

        Wird als Liste gespeichert, damit
        """
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)

        # Hier wird jetzt der L1-Loss zwischen Input und Target berechnet
        self.feat_losses = [self.base_loss(input, target)]

        # hier wird das gleiche nochmal für alle Features gemacht
        self.feat_losses += [self.base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat,
                                                       self.wgts)]
        self.feat_losses += [self.base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat,
                                                       self.wgts)]
        # Wird als Liste gespeichert, um es in metrics abspeichern
        # zu können und printen zu können
        self.metrics = dict(zip(self.metric_names, self.feat_losses))

        # zum Schluss wird hier aufsummiert
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()