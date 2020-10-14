# I don't necessarily want this to be a tf Layer. It has no weights that can be trained
class KbModule:
    """Parent class for tfkbnufft modules.

    This class inherits from nn.Module. It is mostly used to have a central
    location for all __repr__ calls.
    """

    def __repr__(self):
        filter_list = ['interpob', 'buffer', 'parameters', 'hook', 'module']
        tablecheck = False
        out = '\n{}\n'.format(self.__class__.__name__)
        out = out + '----------------------------------------\n'
        for attr, value in self.__dict__.items():
            if 'table' in attr:
                if not tablecheck:
                    out = out + '   table: {} arrays, lengths: {}\n'.format(
                        len(self.table), self.table_oversamp)
                    tablecheck = True
            elif 'traj' in attr or 'scaling_coef' in attr:
                out = out + '   {}: {} {} array\n'.format(
                    attr, value.shape, value.dtype)
            elif not any([item in attr for item in filter_list]):
                out = out + '   {}: {}\n'.format(attr, value)
        return out
