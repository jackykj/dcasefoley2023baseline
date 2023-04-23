



class PixelSNAIL(nn.Module):
    def __init__(
        self,
        shape,
        n_class,  # code nums
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
        cond_embed_channel=1,
        ###
        n_label=7,  # data class nums
        embed_dim=2048,
        ###
    ):
        super().__init__()

        height, width = shape

        self.n_class = n_class

        ###
        self.n_label = n_label
        ###

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size

        self.horizontal = CausalConv2d(
            n_class, channel, [kernel // 2, kernel], padding='down'
        )
        self.vertical = CausalConv2d(
            n_class, channel, [(kernel + 1) // 2, kernel // 2], padding='downright'
        )

        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(
            1, 1, height, width
        )  # shape: torch.Size([1, 1, 20, 86])
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(
            1, 1, height, width
        )  # shape: torch.Size([1, 1, 20, 86])
        # print('x', coord_x.shape, 'y', coord_y.shape)
        self.register_buffer(
            'background', torch.cat([coord_x, coord_y], 1)
        )  # shape: self.background torch.Size([1, 2, 20, 86])

        self.blocks = nn.ModuleList()

        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    channel,
                    res_channel,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition_dim=cond_embed_channel,
                )
            )

        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                n_class, cond_res_channel, cond_res_kernel, n_cond_res_block
            )

        ###
        self.embedNet = EmbedNet(n_label, embed_dim, 20 * 86)
        ###

        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))

        out.extend([nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)])

        self.out = nn.Sequential(*out)

    def forward(self, input, label_condition=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        print('input', input.shape)
        input = (
            F.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self.background)
        )
        print('input', input.shape)
        horizontal = shift_down(self.horizontal(input))
        vertical = shift_right(self.vertical(input))
        out = horizontal + vertical

        # print('background-1', self.background.shape)
        background = self.background[:, :, :height, :].expand(
            batch, 2, height, width
        )  # shape: torch.Size([32, 2, 20, 86])
        # print('background-2', background.shape)
        if True:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]

            else:
                label = F.one_hot(label_condition, self.n_label).type_as(
                    self.background
                )
                # salience = salience_condition.unsqueeze(1)
                # condition = torch.cat((label, salience), 2)
                condition = label

                condition = self.embedNet(condition)
                condition = condition.view(-1, 1, 20, 86)
                # print(condition.shape) #torch.Size([64, 1, 10, 43])
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]
        for block in self.blocks:
            out = block(out, background, condition=condition)  # PixelBlock

        out = self.out(out)

        return out, cache